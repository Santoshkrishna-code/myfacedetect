"""Experiment tracking and management.

Features:
- Track multiple training runs
- Compare experiments
- Manage configurations
- Save results and artifacts
- Resume experiments
"""
import json
import logging
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional, Any
import yaml
import hashlib
from dataclasses import asdict
import pandas as pd
import numpy as np


class ExperimentTracker:
    """Track and manage experiments."""
    
    def __init__(self, experiments_dir: str = 'experiments'):
        """Initialize experiment tracker.
        
        Args:
            experiments_dir: Directory to store experiments
        """
        self.experiments_dir = Path(experiments_dir)
        self.experiments_dir.mkdir(exist_ok=True)
        
        self.metadata_file = self.experiments_dir / 'metadata.json'
        self.experiments = self._load_metadata()
        
        self.logger = logging.getLogger(__name__)
    
    def _load_metadata(self) -> Dict:
        """Load experiments metadata."""
        if self.metadata_file.exists():
            with open(self.metadata_file, 'r') as f:
                return json.load(f)
        return {}
    
    def _save_metadata(self):
        """Save experiments metadata."""
        with open(self.metadata_file, 'w') as f:
            json.dump(self.experiments, f, indent=2)
    
    def create_experiment(self, 
                         name: str,
                         config: Dict,
                         description: str = '') -> str:
        """Create new experiment.
        
        Args:
            name: Experiment name
            config: Configuration dictionary
            description: Experiment description
            
        Returns:
            Experiment ID
        """
        # Generate experiment ID
        exp_id = self._generate_exp_id(name)
        
        # Create experiment directory
        exp_dir = self.experiments_dir / exp_id
        exp_dir.mkdir(exist_ok=True)
        
        # Save configuration
        config_file = exp_dir / 'config.json'
        with open(config_file, 'w') as f:
            json.dump(config, f, indent=2)
        
        # Create experiment metadata
        exp_metadata = {
            'id': exp_id,
            'name': name,
            'description': description,
            'created_at': datetime.now().isoformat(),
            'status': 'created',
            'config_hash': self._hash_config(config),
            'results': None,
            'metrics': {}
        }
        
        self.experiments[exp_id] = exp_metadata
        self._save_metadata()
        
        self.logger.info(f"Experiment created: {exp_id}")
        return exp_id
    
    def _generate_exp_id(self, name: str) -> str:
        """Generate unique experiment ID."""
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        name_short = name.replace(' ', '_')[:20]
        return f"{name_short}_{timestamp}"
    
    def _hash_config(self, config: Dict) -> str:
        """Hash configuration for comparison."""
        config_str = json.dumps(config, sort_keys=True)
        return hashlib.md5(config_str.encode()).hexdigest()[:8]
    
    def start_experiment(self, exp_id: str):
        """Mark experiment as started.
        
        Args:
            exp_id: Experiment ID
        """
        if exp_id in self.experiments:
            self.experiments[exp_id]['status'] = 'running'
            self.experiments[exp_id]['started_at'] = datetime.now().isoformat()
            self._save_metadata()
    
    def log_metrics(self, exp_id: str, epoch: int, **metrics):
        """Log metrics for experiment.
        
        Args:
            exp_id: Experiment ID
            epoch: Epoch number
            **metrics: Metric values
        """
        if exp_id not in self.experiments:
            return
        
        if 'metrics' not in self.experiments[exp_id]:
            self.experiments[exp_id]['metrics'] = {}
        
        if epoch not in self.experiments[exp_id]['metrics']:
            self.experiments[exp_id]['metrics'][epoch] = {}
        
        self.experiments[exp_id]['metrics'][epoch].update(metrics)
        self._save_metadata()
    
    def complete_experiment(self, exp_id: str, results: Dict):
        """Mark experiment as complete.
        
        Args:
            exp_id: Experiment ID
            results: Final results dictionary
        """
        if exp_id not in self.experiments:
            return
        
        exp_dir = self.experiments_dir / exp_id
        
        # Save results
        results_file = exp_dir / 'results.json'
        with open(results_file, 'w') as f:
            json.dump(results, f, indent=2)
        
        self.experiments[exp_id]['status'] = 'completed'
        self.experiments[exp_id]['completed_at'] = datetime.now().isoformat()
        self.experiments[exp_id]['results'] = results
        self._save_metadata()
        
        self.logger.info(f"Experiment completed: {exp_id}")
    
    def get_experiment(self, exp_id: str) -> Optional[Dict]:
        """Get experiment metadata.
        
        Args:
            exp_id: Experiment ID
            
        Returns:
            Experiment metadata or None
        """
        return self.experiments.get(exp_id)
    
    def list_experiments(self, status: Optional[str] = None) -> List[Dict]:
        """List experiments.
        
        Args:
            status: Optional status filter ('created', 'running', 'completed')
            
        Returns:
            List of experiment metadata
        """
        experiments = list(self.experiments.values())
        
        if status:
            experiments = [e for e in experiments if e['status'] == status]
        
        # Sort by creation time, newest first
        experiments.sort(key=lambda x: x['created_at'], reverse=True)
        
        return experiments
    
    def compare_experiments(self, exp_ids: List[str]) -> pd.DataFrame:
        """Compare multiple experiments.
        
        Args:
            exp_ids: List of experiment IDs
            
        Returns:
            DataFrame comparing experiments
        """
        comparison = []
        
        for exp_id in exp_ids:
            exp = self.get_experiment(exp_id)
            if exp is None:
                continue
            
            row = {
                'ID': exp['id'],
                'Name': exp['name'],
                'Status': exp['status'],
                'Created': exp['created_at'],
            }
            
            # Add results if available
            if exp.get('results'):
                row.update(exp['results'])
            
            comparison.append(row)
        
        df = pd.DataFrame(comparison)
        return df
    
    def get_best_experiment(self, metric: str, mode: str = 'max') -> Optional[Dict]:
        """Get best experiment by metric.
        
        Args:
            metric: Metric name to compare
            mode: 'max' or 'min'
            
        Returns:
            Best experiment metadata
        """
        completed = self.list_experiments(status='completed')
        
        if not completed:
            return None
        
        # Get metric values
        experiments_with_metric = []
        for exp in completed:
            if exp.get('results') and metric in exp['results']:
                experiments_with_metric.append(exp)
        
        if not experiments_with_metric:
            return None
        
        # Find best
        if mode == 'max':
            best = max(experiments_with_metric, 
                      key=lambda x: x['results'][metric])
        else:
            best = min(experiments_with_metric,
                      key=lambda x: x['results'][metric])
        
        return best
    
    def export_results(self, output_file: str = 'results_summary.csv'):
        """Export all results to CSV.
        
        Args:
            output_file: Output file path
        """
        completed = self.list_experiments(status='completed')
        
        rows = []
        for exp in completed:
            row = {
                'id': exp['id'],
                'name': exp['name'],
                'created_at': exp['created_at'],
                'completed_at': exp.get('completed_at', 'N/A'),
            }
            
            if exp.get('results'):
                row.update(exp['results'])
            
            rows.append(row)
        
        df = pd.DataFrame(rows)
        df.to_csv(output_file, index=False)
        self.logger.info(f"Results exported to {output_file}")


class CheckpointManager:
    """Manage model checkpoints."""
    
    def __init__(self, checkpoint_dir: str = 'checkpoints'):
        """Initialize checkpoint manager.
        
        Args:
            checkpoint_dir: Directory to store checkpoints
        """
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(exist_ok=True)
        
        self.logger = logging.getLogger(__name__)
    
    def save_checkpoint(self, exp_id: str, checkpoint: Dict, 
                        filename: str = 'model.pt'):
        """Save checkpoint.
        
        Args:
            exp_id: Experiment ID
            checkpoint: Checkpoint dictionary
            filename: Filename
        """
        import torch
        
        exp_dir = self.checkpoint_dir / exp_id
        exp_dir.mkdir(exist_ok=True)
        
        path = exp_dir / filename
        torch.save(checkpoint, path)
        self.logger.info(f"Checkpoint saved: {path}")
    
    def load_checkpoint(self, exp_id: str, filename: str = 'model.pt'):
        """Load checkpoint.
        
        Args:
            exp_id: Experiment ID
            filename: Filename
            
        Returns:
            Checkpoint dictionary
        """
        import torch
        
        path = self.checkpoint_dir / exp_id / filename
        
        if not path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {path}")
        
        checkpoint = torch.load(path)
        self.logger.info(f"Checkpoint loaded: {path}")
        return checkpoint
    
    def get_best_checkpoint(self, exp_id: str) -> Optional[Dict]:
        """Get best checkpoint for experiment.
        
        Args:
            exp_id: Experiment ID
            
        Returns:
            Checkpoint dictionary or None
        """
        try:
            return self.load_checkpoint(exp_id, 'best_model.pt')
        except FileNotFoundError:
            return None
    
    def cleanup_checkpoints(self, exp_id: str, keep_n: int = 3):
        """Keep only the latest N checkpoints.
        
        Args:
            exp_id: Experiment ID
            keep_n: Number of checkpoints to keep
        """
        exp_dir = self.checkpoint_dir / exp_id
        
        if not exp_dir.exists():
            return
        
        # Get all checkpoint files
        checkpoints = sorted(exp_dir.glob('checkpoint_epoch_*.pt'),
                           key=lambda x: x.stat().st_mtime,
                           reverse=True)
        
        # Remove older checkpoints
        for checkpoint in checkpoints[keep_n:]:
            checkpoint.unlink()
            self.logger.info(f"Deleted old checkpoint: {checkpoint}")


if __name__ == '__main__':
    # Example usage
    tracker = ExperimentTracker()
    
    # Create experiment
    config = {
        'model': 'yolov8',
        'batch_size': 32,
        'learning_rate': 1e-3,
        'epochs': 100
    }
    
    exp_id = tracker.create_experiment(
        'yolov8_baseline',
        config,
        'Baseline YOLO v8 detector'
    )
    
    # Start and log metrics
    tracker.start_experiment(exp_id)
    for epoch in range(10):
        tracker.log_metrics(
            exp_id,
            epoch,
            loss=1.0 - epoch * 0.05,
            accuracy=0.5 + epoch * 0.04
        )
    
    # Complete experiment
    tracker.complete_experiment(exp_id, {
        'final_loss': 0.5,
        'final_accuracy': 0.9
    })
    
    # List experiments
    print("All experiments:")
    for exp in tracker.list_experiments():
        print(f"  {exp['id']}: {exp['name']} - {exp['status']}")
