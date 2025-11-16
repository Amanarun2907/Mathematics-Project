"""
ML-Based Difficulty Classifier
==============================

This module uses machine learning to classify Sudoku puzzle difficulty.

MACHINE LEARNING CONCEPTS:
1. Feature extraction from puzzles
2. Classification using Random Forest / Neural Network
3. Pattern recognition for difficulty estimation

FEATURES:
- Number of given cells
- Constraint density
- Option distribution
- Structural patterns

Author: Discrete Mathematics Project
"""

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from typing import Tuple, List, Dict
import pickle
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from solver.constraint_logic import ConstraintChecker
from generator.difficulty_analyzer import DifficultyAnalyzer


class MLDifficultyClassifier:
    """
    Machine learning classifier for Sudoku difficulty
    
    Uses Random Forest for classification based on puzzle features
    """
    
    def __init__(self):
        self.model = RandomForestClassifier(
            n_estimators=100,
            max_depth=10,
            random_state=42
        )
        self.scaler = StandardScaler()
        self.checker = ConstraintChecker()
        self.analyzer = DifficultyAnalyzer()
        self.is_trained = False
        
        # Difficulty mapping
        self.difficulty_map = {
            'Easy': 0,
            'Medium': 1,
            'Hard': 2,
            'Expert': 3
        }
        self.reverse_map = {v: k for k, v in self.difficulty_map.items()}
    
    def extract_features(self, puzzle: np.ndarray) -> np.ndarray:
        """
        Extracts features from Sudoku puzzle
        
        FEATURES:
        1. Given cells count
        2. Empty cells count
        3. Average constraints per cell
        4. Average options per cell
        5. Cells with 1 option
        6. Cells with 2 options
        7. Cells with 3+ options
        8. Row constraint variance
        9. Column constraint variance
        10. Box constraint variance
        
        Args:
            puzzle: 9x9 Sudoku puzzle
        
        Returns:
            Feature vector
        """
        features = []
        
        # Basic counts
        given_cells = np.sum(puzzle != 0)
        empty_cells = np.sum(puzzle == 0)
        features.append(given_cells)
        features.append(empty_cells)
        
        # Constraint and option analysis
        constraints = []
        options = []
        option_counts = {1: 0, 2: 0, 3: 0}
        
        for row in range(9):
            for col in range(9):
                if puzzle[row, col] == 0:
                    # Count constraints
                    constraint_count = self.checker.count_constraints(puzzle, row, col)
                    constraints.append(constraint_count)
                    
                    # Count options
                    valid_options = len(self.checker.get_valid_numbers(puzzle, row, col))
                    options.append(valid_options)
                    
                    # Categorize options
                    if valid_options == 1:
                        option_counts[1] += 1
                    elif valid_options == 2:
                        option_counts[2] += 1
                    else:
                        option_counts[3] += 1
        
        # Statistical features
        if constraints:
            features.append(np.mean(constraints))
            features.append(np.std(constraints))
            features.append(np.min(constraints))
            features.append(np.max(constraints))
        else:
            features.extend([0, 0, 0, 0])
        
        if options:
            features.append(np.mean(options))
            features.append(np.std(options))
            features.append(np.min(options))
            features.append(np.max(options))
        else:
            features.extend([0, 0, 0, 0])
        
        # Option distribution
        features.append(option_counts[1])
        features.append(option_counts[2])
        features.append(option_counts[3])
        
        # Row/Column/Box analysis
        row_constraints = []
        col_constraints = []
        box_constraints = []
        
        for i in range(9):
            row_constraints.append(np.sum(puzzle[i, :] != 0))
            col_constraints.append(np.sum(puzzle[:, i] != 0))
        
        for box_row in range(0, 9, 3):
            for box_col in range(0, 9, 3):
                box = puzzle[box_row:box_row+3, box_col:box_col+3]
                box_constraints.append(np.sum(box != 0))
        
        features.append(np.var(row_constraints))
        features.append(np.var(col_constraints))
        features.append(np.var(box_constraints))
        
        return np.array(features)
    
    def train(self, puzzles: List[np.ndarray], difficulties: List[str]) -> Dict:
        """
        Trains classifier on labeled puzzles
        
        Args:
            puzzles: List of Sudoku puzzles
            difficulties: List of difficulty labels ('Easy', 'Medium', 'Hard', 'Expert')
        
        Returns:
            Training metrics
        """
        # Extract features
        X = np.array([self.extract_features(p) for p in puzzles])
        
        # Encode labels
        y = np.array([self.difficulty_map[d] for d in difficulties])
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        # Train model
        self.model.fit(X_train_scaled, y_train)
        self.is_trained = True
        
        # Evaluate
        train_score = self.model.score(X_train_scaled, y_train)
        test_score = self.model.score(X_test_scaled, y_test)
        
        # Feature importance
        feature_importance = self.model.feature_importances_
        
        metrics = {
            'train_accuracy': train_score,
            'test_accuracy': test_score,
            'feature_importance': feature_importance,
            'n_samples': len(puzzles)
        }
        
        return metrics
    
    def predict(self, puzzle: np.ndarray) -> Tuple[str, np.ndarray]:
        """
        Predicts difficulty of a puzzle
        
        Args:
            puzzle: 9x9 Sudoku puzzle
        
        Returns:
            (difficulty_level, probabilities) tuple
        """
        if not self.is_trained:
            raise ValueError("Model not trained. Call train() first.")
        
        # Extract features
        features = self.extract_features(puzzle).reshape(1, -1)
        features_scaled = self.scaler.transform(features)
        
        # Predict
        prediction = self.model.predict(features_scaled)[0]
        probabilities = self.model.predict_proba(features_scaled)[0]
        
        difficulty = self.reverse_map[prediction]
        
        return difficulty, probabilities
    
    def predict_batch(self, puzzles: List[np.ndarray]) -> List[Tuple[str, np.ndarray]]:
        """
        Predicts difficulty for multiple puzzles
        
        Args:
            puzzles: List of Sudoku puzzles
        
        Returns:
            List of (difficulty, probabilities) tuples
        """
        results = []
        for puzzle in puzzles:
            difficulty, probs = self.predict(puzzle)
            results.append((difficulty, probs))
        return results
    
    def save_model(self, path: str):
        """Saves model to disk"""
        model_data = {
            'model': self.model,
            'scaler': self.scaler,
            'is_trained': self.is_trained
        }
        with open(path, 'wb') as f:
            pickle.dump(model_data, f)
        print(f"Model saved to {path}")
    
    def load_model(self, path: str):
        """Loads model from disk"""
        with open(path, 'rb') as f:
            model_data = pickle.load(f)
        self.model = model_data['model']
        self.scaler = model_data['scaler']
        self.is_trained = model_data['is_trained']
        print(f"Model loaded from {path}")
    
    def get_feature_importance(self) -> Dict[str, float]:
        """
        Returns feature importance scores
        
        Returns:
            Dictionary mapping feature names to importance scores
        """
        if not self.is_trained:
            return {}
        
        feature_names = [
            'given_cells', 'empty_cells',
            'avg_constraints', 'std_constraints', 'min_constraints', 'max_constraints',
            'avg_options', 'std_options', 'min_options', 'max_options',
            'cells_1_option', 'cells_2_options', 'cells_3plus_options',
            'row_variance', 'col_variance', 'box_variance'
        ]
        
        importance = self.model.feature_importances_
        
        return dict(zip(feature_names, importance))


class HybridDifficultyClassifier:
    """
    Combines rule-based and ML-based difficulty classification
    
    Uses both traditional analysis and machine learning for robust classification
    """
    
    def __init__(self):
        self.ml_classifier = MLDifficultyClassifier()
        self.rule_analyzer = DifficultyAnalyzer()
    
    def classify(self, puzzle: np.ndarray, use_ml: bool = True) -> Dict:
        """
        Classifies puzzle difficulty using hybrid approach
        
        Args:
            puzzle: 9x9 Sudoku puzzle
            use_ml: Whether to use ML classifier (if trained)
        
        Returns:
            Dictionary with classification results
        """
        # Rule-based analysis
        rule_metrics = self.rule_analyzer.analyze_difficulty(puzzle)
        
        result = {
            'rule_based': {
                'difficulty': rule_metrics['difficulty_level'],
                'score': rule_metrics['difficulty_score']
            }
        }
        
        # ML-based classification (if available)
        if use_ml and self.ml_classifier.is_trained:
            ml_difficulty, ml_probs = self.ml_classifier.predict(puzzle)
            result['ml_based'] = {
                'difficulty': ml_difficulty,
                'probabilities': {
                    'Easy': ml_probs[0],
                    'Medium': ml_probs[1],
                    'Hard': ml_probs[2],
                    'Expert': ml_probs[3]
                }
            }
            
            # Consensus
            if rule_metrics['difficulty_level'] == ml_difficulty:
                result['consensus'] = ml_difficulty
                result['confidence'] = 'High'
            else:
                # Use ML prediction with lower confidence
                result['consensus'] = ml_difficulty
                result['confidence'] = 'Medium'
        else:
            result['consensus'] = rule_metrics['difficulty_level']
            result['confidence'] = 'Rule-based only'
        
        return result


if __name__ == "__main__":
    print("ML Difficulty Classifier")
    print("=" * 60)
    
    # Example puzzle
    puzzle = np.array([
        [5, 3, 0, 0, 7, 0, 0, 0, 0],
        [6, 0, 0, 1, 9, 5, 0, 0, 0],
        [0, 9, 8, 0, 0, 0, 0, 6, 0],
        [8, 0, 0, 0, 6, 0, 0, 0, 3],
        [4, 0, 0, 8, 0, 3, 0, 0, 1],
        [7, 0, 0, 0, 2, 0, 0, 0, 6],
        [0, 6, 0, 0, 0, 0, 2, 8, 0],
        [0, 0, 0, 4, 1, 9, 0, 0, 5],
        [0, 0, 0, 0, 8, 0, 0, 7, 9]
    ])
    
    classifier = MLDifficultyClassifier()
    
    print("\nExtracting features...")
    features = classifier.extract_features(puzzle)
    print(f"Feature vector shape: {features.shape}")
    print(f"Features: {features}")
    
    print("\nNote: Model needs training data to make predictions.")
    print("Generate labeled puzzles and call train() method.")
