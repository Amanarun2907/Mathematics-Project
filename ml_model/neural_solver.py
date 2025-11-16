"""
Neural Network Sudoku Solver
============================

This module implements a deep learning approach to Sudoku solving.

MACHINE LEARNING CONCEPTS:
1. Convolutional Neural Networks (CNN) - Pattern recognition
2. Multi-class classification - Predicting numbers 1-9
3. Training on large datasets
4. Feature extraction from grid patterns

ARCHITECTURE:
- Input: 9x9x10 tensor (9 digits + empty indicator)
- Multiple Conv2D layers for pattern recognition
- Dense layers for classification
- Output: 9x9x9 tensor (probability distribution for each cell)

Author: Discrete Mathematics Project
"""

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models
from typing import Tuple, List
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from solver.backtracking_solver import BacktrackingSolver
from solver.constraint_logic import ConstraintChecker


class NeuralSudokuSolver:
    """
    Deep learning-based Sudoku solver
    
    Uses CNN to learn patterns and predict cell values
    """
    
    def __init__(self, model_path: str = None):
        """
        Initialize neural solver
        
        Args:
            model_path: Path to saved model (optional)
        """
        self.model = None
        self.checker = ConstraintChecker()
        self.backtrack_solver = BacktrackingSolver()
        
        if model_path and os.path.exists(model_path):
            self.load_model(model_path)
        else:
            self.model = self._build_model()
    
    def _build_model(self) -> keras.Model:
        """
        Builds CNN architecture for Sudoku solving
        
        ARCHITECTURE:
        - Input: 9x9x10 (one-hot encoded + empty indicator)
        - Conv2D layers: Extract spatial patterns
        - Dense layers: Classification
        - Output: 9x9x9 (probability for each digit)
        """
        # Input: 9x9 grid with 10 channels (digits 1-9 + empty)
        input_layer = layers.Input(shape=(9, 9, 10))
        
        # Convolutional layers for pattern recognition
        # DISCRETE MATH: Learns constraint patterns through convolution
        x = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(input_layer)
        x = layers.BatchNormalization()(x)
        x = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(x)
        x = layers.BatchNormalization()(x)
        
        x = layers.Conv2D(128, (3, 3), activation='relu', padding='same')(x)
        x = layers.BatchNormalization()(x)
        x = layers.Conv2D(128, (3, 3), activation='relu', padding='same')(x)
        x = layers.BatchNormalization()(x)
        
        x = layers.Conv2D(256, (3, 3), activation='relu', padding='same')(x)
        x = layers.BatchNormalization()(x)
        
        # Output layer: 9 channels for 9 possible digits
        output = layers.Conv2D(9, (1, 1), activation='softmax', padding='same')(x)
        
        model = models.Model(inputs=input_layer, outputs=output)
        
        # Compile model
        model.compile(
            optimizer='adam',
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )
        
        return model
    
    def _encode_board(self, board: np.ndarray) -> np.ndarray:
        """
        Encodes board into one-hot representation
        
        ENCODING:
        - 10 channels: digits 1-9 + empty indicator
        - Channel 0: empty cells (1 if empty, 0 otherwise)
        - Channels 1-9: one-hot encoding of digits
        
        Args:
            board: 9x9 Sudoku board
        
        Returns:
            9x9x10 encoded tensor
        """
        encoded = np.zeros((9, 9, 10), dtype=np.float32)
        
        for row in range(9):
            for col in range(9):
                if board[row, col] == 0:
                    encoded[row, col, 0] = 1  # Empty indicator
                else:
                    digit = int(board[row, col])
                    encoded[row, col, digit] = 1  # One-hot encoding
        
        return encoded
    
    def _decode_prediction(self, prediction: np.ndarray, original_board: np.ndarray) -> np.ndarray:
        """
        Decodes model prediction into Sudoku board
        
        Args:
            prediction: 9x9x9 probability tensor
            original_board: Original puzzle (to preserve given cells)
        
        Returns:
            9x9 Sudoku board
        """
        board = original_board.copy()
        
        # For each empty cell, take argmax of probabilities
        for row in range(9):
            for col in range(9):
                if original_board[row, col] == 0:
                    # Get predicted digit (1-9)
                    predicted_digit = np.argmax(prediction[row, col]) + 1
                    board[row, col] = predicted_digit
        
        return board
    
    def solve(self, puzzle: np.ndarray, use_constraints: bool = True) -> Tuple[bool, np.ndarray]:
        """
        Solves Sudoku using neural network
        
        Args:
            puzzle: 9x9 Sudoku puzzle
            use_constraints: Whether to apply constraint checking
        
        Returns:
            (success, solved_board) tuple
        """
        if self.model is None:
            raise ValueError("Model not loaded. Train or load a model first.")
        
        # Encode puzzle
        encoded = self._encode_board(puzzle)
        encoded = np.expand_dims(encoded, axis=0)  # Add batch dimension
        
        # Predict
        prediction = self.model.predict(encoded, verbose=0)[0]
        
        # Decode prediction
        solved = self._decode_prediction(prediction, puzzle)
        
        # Apply constraint checking if requested
        if use_constraints:
            solved = self._apply_constraints(solved, puzzle)
        
        # Validate solution
        is_valid = self.checker.is_solved(solved)
        
        return is_valid, solved
    
    def _apply_constraints(self, board: np.ndarray, original: np.ndarray) -> np.ndarray:
        """
        Applies constraint checking to neural network output
        
        HYBRID APPROACH: Combines ML prediction with logical constraints
        
        Args:
            board: Predicted board
            original: Original puzzle
        
        Returns:
            Corrected board
        """
        corrected = board.copy()
        
        # For each predicted cell, verify it satisfies constraints
        for row in range(9):
            for col in range(9):
                if original[row, col] == 0:
                    predicted = corrected[row, col]
                    
                    # Check if prediction is valid
                    if not self.checker.is_valid_placement(corrected, row, col, predicted):
                        # If invalid, try to find valid number
                        valid_numbers = self.checker.get_valid_numbers(corrected, row, col)
                        if len(valid_numbers) > 0:
                            corrected[row, col] = list(valid_numbers)[0]
                        else:
                            corrected[row, col] = 0  # Mark as unsolved
        
        return corrected
    
    def solve_iterative(self, puzzle: np.ndarray, max_iterations: int = 10) -> Tuple[bool, np.ndarray]:
        """
        Solves puzzle iteratively using neural network
        
        ITERATIVE APPROACH:
        1. Predict all cells
        2. Fill high-confidence predictions
        3. Repeat until solved or max iterations
        
        Args:
            puzzle: 9x9 Sudoku puzzle
            max_iterations: Maximum iterations
        
        Returns:
            (success, solved_board) tuple
        """
        current = puzzle.copy()
        
        for iteration in range(max_iterations):
            # Encode and predict
            encoded = self._encode_board(current)
            encoded = np.expand_dims(encoded, axis=0)
            prediction = self.model.predict(encoded, verbose=0)[0]
            
            # Fill high-confidence predictions
            filled_any = False
            for row in range(9):
                for col in range(9):
                    if current[row, col] == 0:
                        # Get confidence scores
                        confidences = prediction[row, col]
                        max_confidence = np.max(confidences)
                        predicted_digit = np.argmax(confidences) + 1
                        
                        # Fill if high confidence and valid
                        if max_confidence > 0.9:
                            if self.checker.is_valid_placement(current, row, col, predicted_digit):
                                current[row, col] = predicted_digit
                                filled_any = True
            
            # Check if solved
            if self.checker.is_solved(current):
                return True, current
            
            # If no progress, break
            if not filled_any:
                break
        
        # If not solved, use backtracking as fallback
        success, solved = self.backtrack_solver.solve(current)
        return success, solved
    
    def train(self, puzzles: np.ndarray, solutions: np.ndarray, 
              epochs: int = 10, batch_size: int = 32, validation_split: float = 0.2):
        """
        Trains the neural network on Sudoku puzzles
        
        Args:
            puzzles: Array of Sudoku puzzles (N, 9, 9)
            solutions: Array of solutions (N, 9, 9)
            epochs: Number of training epochs
            batch_size: Batch size
            validation_split: Validation data fraction
        
        Returns:
            Training history
        """
        # Encode puzzles and solutions
        X = np.array([self._encode_board(p) for p in puzzles])
        
        # Encode solutions as one-hot (9x9x9)
        y = np.zeros((len(solutions), 9, 9, 9), dtype=np.float32)
        for i, solution in enumerate(solutions):
            for row in range(9):
                for col in range(9):
                    digit = int(solution[row, col]) - 1  # 0-indexed
                    y[i, row, col, digit] = 1
        
        # Train model
        history = self.model.fit(
            X, y,
            epochs=epochs,
            batch_size=batch_size,
            validation_split=validation_split,
            verbose=1
        )
        
        return history
    
    def save_model(self, path: str):
        """Saves model to disk"""
        if self.model:
            self.model.save(path)
            print(f"Model saved to {path}")
    
    def load_model(self, path: str):
        """Loads model from disk"""
        self.model = keras.models.load_model(path)
        print(f"Model loaded from {path}")
    
    def get_model_summary(self) -> str:
        """Returns model architecture summary"""
        if self.model:
            from io import StringIO
            stream = StringIO()
            self.model.summary(print_fn=lambda x: stream.write(x + '\n'))
            return stream.getvalue()
        return "No model loaded"


if __name__ == "__main__":
    print("Neural Sudoku Solver")
    print("=" * 60)
    
    # Create solver
    solver = NeuralSudokuSolver()
    
    print("\nModel Architecture:")
    print(solver.get_model_summary())
    
    print("\nNote: Model needs to be trained on large dataset for good performance.")
    print("For demonstration, the model is initialized but not trained.")
    print("In production, train on 100K+ puzzle-solution pairs.")
