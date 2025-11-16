"""
Model Training Utilities
========================

This module provides utilities for training ML models on Sudoku data.

FUNCTIONALITY:
1. Dataset generation
2. Model training
3. Evaluation and metrics
4. Model persistence

Author: Discrete Mathematics Project
"""

import numpy as np
from typing import List, Tuple, Dict
import time
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from generator.puzzle_generator import PuzzleGenerator
from generator.difficulty_analyzer import DifficultyAnalyzer
from ml_model.neural_solver import NeuralSudokuSolver
from ml_model.difficulty_classifier import MLDifficultyClassifier


class DatasetGenerator:
    """
    Generates training datasets for ML models
    """
    
    def __init__(self, seed: int = 42):
        self.generator = PuzzleGenerator(seed=seed)
        self.analyzer = DifficultyAnalyzer()
    
    def generate_solver_dataset(self, n_samples: int = 1000) -> Tuple[np.ndarray, np.ndarray]:
        """
        Generates dataset for training neural solver
        
        Args:
            n_samples: Number of puzzle-solution pairs
        
        Returns:
            (puzzles, solutions) tuple
        """
        print(f"Generating {n_samples} puzzle-solution pairs...")
        
        puzzles = []
        solutions = []
        
        difficulties = ['easy', 'medium', 'hard', 'expert']
        
        for i in range(n_samples):
            if i % 100 == 0:
                print(f"Progress: {i}/{n_samples}")
            
            # Generate puzzle with random difficulty
            difficulty = np.random.choice(difficulties)
            puzzle, solution = self.generator.generate_puzzle(difficulty)
            
            puzzles.append(puzzle)
            solutions.append(solution)
        
        print("Dataset generation complete!")
        
        return np.array(puzzles), np.array(solutions)
    
    def generate_classifier_dataset(self, n_per_difficulty: int = 250) -> Tuple[List[np.ndarray], List[str]]:
        """
        Generates labeled dataset for difficulty classifier
        
        Args:
            n_per_difficulty: Number of puzzles per difficulty level
        
        Returns:
            (puzzles, labels) tuple
        """
        print(f"Generating {n_per_difficulty * 4} labeled puzzles...")
        
        puzzles = []
        labels = []
        
        difficulties = ['easy', 'medium', 'hard', 'expert']
        
        for difficulty in difficulties:
            print(f"Generating {difficulty} puzzles...")
            
            for i in range(n_per_difficulty):
                puzzle, _ = self.generator.generate_puzzle(difficulty)
                
                # Verify difficulty with analyzer
                analysis = self.analyzer.analyze_difficulty(puzzle)
                actual_difficulty = analysis['difficulty_level']
                
                puzzles.append(puzzle)
                labels.append(actual_difficulty)
        
        print("Dataset generation complete!")
        
        return puzzles, labels
    
    def save_dataset(self, puzzles: np.ndarray, solutions: np.ndarray, path: str):
        """Saves dataset to disk"""
        np.savez_compressed(path, puzzles=puzzles, solutions=solutions)
        print(f"Dataset saved to {path}")
    
    def load_dataset(self, path: str) -> Tuple[np.ndarray, np.ndarray]:
        """Loads dataset from disk"""
        data = np.load(path)
        return data['puzzles'], data['solutions']


class ModelTrainer:
    """
    Trains and evaluates ML models
    """
    
    def __init__(self):
        self.dataset_gen = DatasetGenerator()
    
    def train_neural_solver(self, n_samples: int = 1000, epochs: int = 10, 
                           model_save_path: str = None) -> Dict:
        """
        Trains neural network solver
        
        Args:
            n_samples: Number of training samples
            epochs: Training epochs
            model_save_path: Path to save trained model
        
        Returns:
            Training metrics
        """
        print("=" * 60)
        print("TRAINING NEURAL SUDOKU SOLVER")
        print("=" * 60)
        
        # Generate dataset
        start_time = time.time()
        puzzles, solutions = self.dataset_gen.generate_solver_dataset(n_samples)
        dataset_time = time.time() - start_time
        
        print(f"\nDataset generated in {dataset_time:.2f} seconds")
        print(f"Puzzles shape: {puzzles.shape}")
        print(f"Solutions shape: {solutions.shape}")
        
        # Create and train model
        print("\nInitializing neural network...")
        solver = NeuralSudokuSolver()
        
        print("\nTraining model...")
        train_start = time.time()
        history = solver.train(puzzles, solutions, epochs=epochs, batch_size=32)
        train_time = time.time() - train_start
        
        print(f"\nTraining completed in {train_time:.2f} seconds")
        
        # Save model if path provided
        if model_save_path:
            solver.save_model(model_save_path)
        
        # Compile metrics
        metrics = {
            'n_samples': n_samples,
            'epochs': epochs,
            'dataset_generation_time': dataset_time,
            'training_time': train_time,
            'final_loss': float(history.history['loss'][-1]),
            'final_accuracy': float(history.history['accuracy'][-1]),
            'final_val_loss': float(history.history['val_loss'][-1]),
            'final_val_accuracy': float(history.history['val_accuracy'][-1])
        }
        
        return metrics
    
    def train_difficulty_classifier(self, n_per_difficulty: int = 250,
                                   model_save_path: str = None) -> Dict:
        """
        Trains difficulty classifier
        
        Args:
            n_per_difficulty: Samples per difficulty level
            model_save_path: Path to save trained model
        
        Returns:
            Training metrics
        """
        print("=" * 60)
        print("TRAINING DIFFICULTY CLASSIFIER")
        print("=" * 60)
        
        # Generate dataset
        start_time = time.time()
        puzzles, labels = self.dataset_gen.generate_classifier_dataset(n_per_difficulty)
        dataset_time = time.time() - start_time
        
        print(f"\nDataset generated in {dataset_time:.2f} seconds")
        print(f"Total samples: {len(puzzles)}")
        
        # Create and train classifier
        print("\nTraining classifier...")
        classifier = MLDifficultyClassifier()
        
        train_start = time.time()
        metrics = classifier.train(puzzles, labels)
        train_time = time.time() - train_start
        
        print(f"\nTraining completed in {train_time:.2f} seconds")
        print(f"Train accuracy: {metrics['train_accuracy']:.4f}")
        print(f"Test accuracy: {metrics['test_accuracy']:.4f}")
        
        # Save model if path provided
        if model_save_path:
            classifier.save_model(model_save_path)
        
        # Add timing to metrics
        metrics['dataset_generation_time'] = dataset_time
        metrics['training_time'] = train_time
        
        # Feature importance
        print("\nTop 5 Important Features:")
        importance = classifier.get_feature_importance()
        sorted_features = sorted(importance.items(), key=lambda x: x[1], reverse=True)
        for feature, score in sorted_features[:5]:
            print(f"  {feature}: {score:.4f}")
        
        return metrics
    
    def evaluate_neural_solver(self, model_path: str, n_test: int = 100) -> Dict:
        """
        Evaluates trained neural solver
        
        Args:
            model_path: Path to trained model
            n_test: Number of test puzzles
        
        Returns:
            Evaluation metrics
        """
        print("=" * 60)
        print("EVALUATING NEURAL SOLVER")
        print("=" * 60)
        
        # Load model
        solver = NeuralSudokuSolver(model_path)
        
        # Generate test puzzles
        print(f"\nGenerating {n_test} test puzzles...")
        puzzles, solutions = self.dataset_gen.generate_solver_dataset(n_test)
        
        # Evaluate
        print("\nEvaluating...")
        correct = 0
        total_time = 0
        
        for i, (puzzle, solution) in enumerate(zip(puzzles, solutions)):
            start = time.time()
            success, predicted = solver.solve(puzzle)
            solve_time = time.time() - start
            
            total_time += solve_time
            
            if success and np.array_equal(predicted, solution):
                correct += 1
            
            if (i + 1) % 10 == 0:
                print(f"Progress: {i + 1}/{n_test}")
        
        accuracy = correct / n_test
        avg_time = total_time / n_test
        
        metrics = {
            'accuracy': accuracy,
            'correct': correct,
            'total': n_test,
            'avg_solve_time': avg_time,
            'total_time': total_time
        }
        
        print(f"\nAccuracy: {accuracy:.2%}")
        print(f"Average solve time: {avg_time:.4f} seconds")
        
        return metrics
    
    def evaluate_difficulty_classifier(self, model_path: str, n_test: int = 100) -> Dict:
        """
        Evaluates trained difficulty classifier
        
        Args:
            model_path: Path to trained model
            n_test: Number of test puzzles per difficulty
        
        Returns:
            Evaluation metrics
        """
        print("=" * 60)
        print("EVALUATING DIFFICULTY CLASSIFIER")
        print("=" * 60)
        
        # Load model
        classifier = MLDifficultyClassifier()
        classifier.load_model(model_path)
        
        # Generate test data
        print(f"\nGenerating test data...")
        puzzles, labels = self.dataset_gen.generate_classifier_dataset(n_test // 4)
        
        # Evaluate
        print("\nEvaluating...")
        predictions = classifier.predict_batch(puzzles)
        
        correct = 0
        confusion = {level: {level: 0 for level in ['Easy', 'Medium', 'Hard', 'Expert']} 
                    for level in ['Easy', 'Medium', 'Hard', 'Expert']}
        
        for (pred, _), true in zip(predictions, labels):
            if pred == true:
                correct += 1
            confusion[true][pred] += 1
        
        accuracy = correct / len(puzzles)
        
        metrics = {
            'accuracy': accuracy,
            'correct': correct,
            'total': len(puzzles),
            'confusion_matrix': confusion
        }
        
        print(f"\nAccuracy: {accuracy:.2%}")
        print("\nConfusion Matrix:")
        print("True \\ Pred | Easy | Medium | Hard | Expert")
        print("-" * 50)
        for true_label in ['Easy', 'Medium', 'Hard', 'Expert']:
            row = f"{true_label:12} |"
            for pred_label in ['Easy', 'Medium', 'Hard', 'Expert']:
                row += f" {confusion[true_label][pred_label]:4} |"
            print(row)
        
        return metrics


if __name__ == "__main__":
    print("Model Training Utilities")
    print("=" * 60)
    
    trainer = ModelTrainer()
    
    print("\nThis module provides training utilities for:")
    print("1. Neural Sudoku Solver")
    print("2. Difficulty Classifier")
    print("\nUse the train_* methods to train models.")
    print("Example: trainer.train_difficulty_classifier(n_per_difficulty=100)")
