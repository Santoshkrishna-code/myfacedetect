"""Hyperparameter tuning for model optimization using Optuna.

Features:
- Bayesian optimization
- Multi-objective optimization
- Parallel trials
- Result visualization
- Best model saving
"""
import optuna
from optuna.visualization import plot_optimization_history, plot_param_importances
import json
import os
from pathlib import Path
from typing import Dict, Callable, Tuple, Optional, List
import numpy as np


class HyperparameterTuner:
    """Hyperparameter optimization using Optuna."""
    
    def __init__(self, study_name: str = "face_detection", storage: Optional[str] = None):
        """Initialize tuner.
        
        Args:
            study_name: Name of the study
            storage: Optional database URL for persistence
        """
        self.study_name = study_name
        self.storage = storage or f'sqlite:///{study_name}.db'
        
        # Create study
        self.study = optuna.create_study(
            study_name=study_name,
            storage=self.storage,
            load_if_exists=True,
            direction='maximize',  # Maximize mAP
            sampler=optuna.samplers.TPESampler(seed=42)
        )
        
        self.best_trial = None
        self.results_history = []
    
    def suggest_hyperparameters(self, trial: optuna.Trial) -> Dict:
        """Suggest hyperparameters for a trial.
        
        Args:
            trial: Optuna trial object
            
        Returns:
            Dictionary of suggested hyperparameters
        """
        params = {
            # Learning rate
            'lr': trial.suggest_float('lr', 1e-5, 1e-2, log=True),
            'lr_scheduler': trial.suggest_categorical('lr_scheduler', ['cosine', 'linear', 'step']),
            
            # Batch size
            'batch_size': trial.suggest_categorical('batch_size', [8, 16, 32, 64]),
            
            # Optimizer
            'optimizer': trial.suggest_categorical('optimizer', ['adam', 'sgd', 'adamw']),
            'momentum': trial.suggest_float('momentum', 0.5, 0.99),
            'weight_decay': trial.suggest_float('weight_decay', 0, 1e-3),
            
            # Model
            'model_size': trial.suggest_categorical('model_size', ['nano', 'small', 'medium', 'large']),
            
            # Augmentation
            'augmentation_intensity': trial.suggest_float('augmentation_intensity', 0.0, 1.0),
            'mixup_prob': trial.suggest_float('mixup_prob', 0.0, 0.5),
            'mosaic_prob': trial.suggest_float('mosaic_prob', 0.0, 0.5),
            
            # Training
            'dropout': trial.suggest_float('dropout', 0.0, 0.5),
            'label_smoothing': trial.suggest_float('label_smoothing', 0.0, 0.2),
            'warmup_epochs': trial.suggest_int('warmup_epochs', 0, 10),
        }
        
        return params
    
    def objective(self, trial: optuna.Trial, training_func: Callable) -> float:
        """Objective function for optimization.
        
        Args:
            trial: Optuna trial
            training_func: Function that trains model and returns mAP
            
        Returns:
            Metric value to optimize (mAP)
        """
        params = self.suggest_hyperparameters(trial)
        
        try:
            # Call training function with parameters
            metric = training_func(params, trial)
            
            # Save result
            self.results_history.append({
                'trial': trial.number,
                'params': params,
                'metric': metric
            })
            
            return metric
            
        except Exception as e:
            print(f"Trial {trial.number} failed: {e}")
            # Penalize failed trials
            return -1.0
    
    def optimize(self, training_func: Callable, n_trials: int = 100, n_jobs: int = 1):
        """Run optimization.
        
        Args:
            training_func: Training function that returns metric
            n_trials: Number of trials
            n_jobs: Number of parallel jobs
        """
        print(f"Starting hyperparameter optimization: {n_trials} trials")
        
        self.study.optimize(
            lambda trial: self.objective(trial, training_func),
            n_trials=n_trials,
            n_jobs=n_jobs,
            show_progress_bar=True
        )
        
        self.best_trial = self.study.best_trial
        
        print(f"\n✅ Optimization complete!")
        print(f"Best trial: {self.best_trial.number}")
        print(f"Best metric: {self.best_trial.value:.4f}")
        print(f"Best parameters:")
        for key, value in self.best_trial.params.items():
            print(f"  - {key}: {value}")
    
    def get_best_params(self) -> Dict:
        """Get best hyperparameters found.
        
        Returns:
            Dictionary of best parameters
        """
        if self.best_trial is None:
            self.best_trial = self.study.best_trial
        
        return self.best_trial.params
    
    def save_results(self, output_path: str = 'tuning_results.json'):
        """Save optimization results.
        
        Args:
            output_path: Path to save results JSON
        """
        results = {
            'study_name': self.study_name,
            'n_trials': len(self.study.trials),
            'best_trial': {
                'number': self.best_trial.number,
                'value': float(self.best_trial.value),
                'params': self.best_trial.params
            },
            'trials_history': self.results_history
        }
        
        with open(output_path, 'w') as f:
            json.dump(results, f, indent=2)
        
        print(f"Results saved to {output_path}")
        
        return results
    
    def visualize_results(self, output_dir: str = 'tuning_plots'):
        """Generate visualization plots.
        
        Args:
            output_dir: Directory to save plots
        """
        os.makedirs(output_dir, exist_ok=True)
        
        # Optimization history
        fig = plot_optimization_history(self.study)
        fig.write_html(f'{output_dir}/optimization_history.html')
        
        # Parameter importances
        fig = plot_param_importances(self.study)
        fig.write_html(f'{output_dir}/param_importances.html')
        
        print(f"Plots saved to {output_dir}/")


class TrialCallback:
    """Callback for monitoring trials during optimization."""
    
    def __init__(self, output_dir: str = 'trial_logs'):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
    
    def __call__(self, study: optuna.Study, trial: optuna.Trial) -> None:
        """Called after each trial completes.
        
        Args:
            study: Optuna study object
            trial: Completed trial
        """
        log_data = {
            'trial_number': trial.number,
            'value': trial.value,
            'params': trial.params,
            'state': trial.state.name
        }
        
        log_path = self.output_dir / f'trial_{trial.number}.json'
        with open(log_path, 'w') as f:
            json.dump(log_data, f, indent=2)


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='Hyperparameter tuning')
    parser.add_argument('--n-trials', type=int, default=50, help='Number of trials')
    parser.add_argument('--study-name', default='face_detection', help='Study name')
    parser.add_argument('--output-dir', default='tuning_results', help='Output directory')
    
    args = parser.parse_args()
    
    def dummy_training_func(params: Dict, trial: optuna.Trial) -> float:
        """Example training function that returns a metric."""
        # In practice, this would train a model and return mAP
        import random
        return random.uniform(0.7, 0.95)
    
    tuner = HyperparameterTuner(study_name=args.study_name)
    tuner.optimize(dummy_training_func, n_trials=args.n_trials)
    
    results = tuner.save_results(f'{args.output_dir}/results.json')
    tuner.visualize_results(args.output_dir)
