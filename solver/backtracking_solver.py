"""
Backtracking Algorithm for Sudoku Solving
=========================================

This module implements the classical backtracking algorithm using discrete mathematics concepts.

DISCRETE MATH CONCEPTS:
1. Backtracking - Recursive depth-first search with pruning
2. Recursion - Function calling itself with smaller subproblems
3. State Space Tree - Tree of all possible configurations
4. Pruning - Eliminating invalid branches early

ALGORITHM:
1. Find empty cell
2. Try numbers 1-9
3. If valid, place number and recurse
4. If recursion succeeds, solution found
5. If recursion fails, backtrack (undo placement)
6. If all numbers fail, return False (dead end)

TIME COMPLEXITY: O(9^(n*n)) worst case, where n=9
SPACE COMPLEXITY: O(n*n) for recursion stack

Author: Discrete Mathematics Project
"""

import numpy as np
from typing import Tuple, Optional, List
import time
import sys
import os

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from solver.constraint_logic import ConstraintChecker, ConstraintPropagation


class BacktrackingSolver:
    """
    Implements backtracking algorithm for Sudoku solving
    
    DISCRETE MATH: Backtracking is a form of exhaustive search
    that uses depth-first traversal of the state space tree
    """
    
    def __init__(self):
        self.checker = ConstraintChecker()
        self.propagator = ConstraintPropagation()
        
        # Statistics tracking
        self.recursion_count = 0
        self.backtrack_count = 0
        self.steps = []
        self.solving_time = 0.0
    
    def solve(self, board: np.ndarray, use_propagation: bool = True) -> Tuple[bool, np.ndarray]:
        """
        Solves Sudoku puzzle using backtracking
        
        DISCRETE MATH CONCEPT: Recursive backtracking
        - Base case: Board is complete and valid
        - Recursive case: Try each valid number, recurse
        - Backtrack: If recursion fails, undo and try next number
        
        Args:
            board: 9x9 Sudoku board (0 for empty cells)
            use_propagation: Whether to use constraint propagation first
        
        Returns:
            (success, solved_board) tuple
        """
        # Reset statistics
        self.recursion_count = 0
        self.backtrack_count = 0
        self.steps = []
        
        # Make a copy to avoid modifying original
        board_copy = board.copy()
        
        # Start timing
        start_time = time.time()
        
        # Apply constraint propagation first (optimization)
        if use_propagation:
            filled = self.propagator.propagate_constraints(board_copy)
            if filled > 0:
                self.steps.append(('propagation', filled))
        
        # Apply backtracking
        success = self._backtrack(board_copy)
        
        # End timing
        self.solving_time = time.time() - start_time
        
        return success, board_copy
    
    def _backtrack(self, board: np.ndarray) -> bool:
        """
        Recursive backtracking function
        
        DISCRETE MATH CONCEPT: Recursion with backtracking
        
        Algorithm:
        1. BASE CASE: If board complete, return True
        2. RECURSIVE CASE:
           a. Find empty cell
           b. For each number 1-9:
              - If valid, place number
              - Recurse on new board state
              - If recursion succeeds, return True
              - Else, backtrack (remove number)
           c. If all numbers fail, return False
        
        Args:
            board: Current board state
        
        Returns:
            True if solution found, False otherwise
        """
        # Increment recursion counter
        self.recursion_count += 1
        
        # BASE CASE: Check if board is complete
        if self.checker.is_board_complete(board):
            return True
        
        # RECURSIVE CASE: Find empty cell
        empty_cell = self._find_empty_cell(board)
        if empty_cell is None:
            return False
        
        row, col = empty_cell
        
        # Get valid numbers for this cell (optimization using constraint logic)
        valid_numbers = self.checker.get_valid_numbers(board, row, col)
        
        # Try each valid number
        for num in valid_numbers:
            # Place number (DISCRETE MATH: State transition)
            board[row, col] = num
            self.steps.append(('place', row, col, num))
            
            # RECURSE: Try to solve with this placement
            if self._backtrack(board):
                return True  # Solution found!
            
            # BACKTRACK: Placement didn't lead to solution
            board[row, col] = 0
            self.backtrack_count += 1
            self.steps.append(('backtrack', row, col, num))
        
        # All numbers failed - dead end
        return False
    
    def _find_empty_cell(self, board: np.ndarray) -> Optional[Tuple[int, int]]:
        """
        Finds next empty cell to fill
        
        HEURISTIC: Choose cell with minimum remaining values (MRV)
        This is an optimization that reduces backtracking
        
        DISCRETE MATH: Greedy heuristic for search optimization
        
        Returns:
            (row, col) of empty cell, or None if board complete
        """
        min_options = 10
        best_cell = None
        
        for row in range(9):
            for col in range(9):
                if board[row, col] == 0:
                    # Count valid options for this cell
                    valid = self.checker.get_valid_numbers(board, row, col)
                    num_options = len(valid)
                    
                    # If no options, this is a dead end
                    if num_options == 0:
                        return (row, col)
                    
                    # Choose cell with minimum options (MRV heuristic)
                    if num_options < min_options:
                        min_options = num_options
                        best_cell = (row, col)
        
        return best_cell
    
    def get_statistics(self) -> dict:
        """
        Returns solving statistics
        
        Returns:
            Dictionary with performance metrics
        """
        return {
            'recursion_count': self.recursion_count,
            'backtrack_count': self.backtrack_count,
            'total_steps': len(self.steps),
            'solving_time': self.solving_time,
            'backtracks_per_second': self.backtrack_count / self.solving_time if self.solving_time > 0 else 0
        }
    
    def solve_with_steps(self, board: np.ndarray) -> Tuple[bool, np.ndarray, List]:
        """
        Solves puzzle and returns step-by-step solution
        
        Useful for visualization and educational purposes
        
        Returns:
            (success, solved_board, steps) tuple
        """
        success, solved = self.solve(board)
        return success, solved, self.steps


class OptimizedBacktrackingSolver(BacktrackingSolver):
    """
    Enhanced backtracking solver with advanced optimizations
    
    OPTIMIZATIONS:
    1. Constraint propagation before backtracking
    2. Minimum Remaining Values (MRV) heuristic
    3. Forward checking
    4. Naked singles and hidden singles
    """
    
    def __init__(self):
        super().__init__()
    
    def _backtrack(self, board: np.ndarray) -> bool:
        """
        Optimized backtracking with constraint propagation at each step
        
        DISCRETE MATH: Combines backtracking with constraint satisfaction
        """
        self.recursion_count += 1
        
        # Apply constraint propagation at each step
        filled = self.propagator.propagate_constraints(board, max_iterations=1)
        if filled > 0:
            self.steps.append(('propagation', filled))
        
        # Check if solved
        if self.checker.is_board_complete(board):
            return self.checker.is_board_valid(board)
        
        # Find empty cell with MRV heuristic
        empty_cell = self._find_empty_cell(board)
        if empty_cell is None:
            return False
        
        row, col = empty_cell
        valid_numbers = self.checker.get_valid_numbers(board, row, col)
        
        # If no valid numbers, dead end
        if len(valid_numbers) == 0:
            return False
        
        # Try each valid number
        for num in valid_numbers:
            board[row, col] = num
            self.steps.append(('place', row, col, num))
            
            # Check if this creates an unsolvable state (forward checking)
            if self._is_solvable(board):
                if self._backtrack(board):
                    return True
            
            # Backtrack
            board[row, col] = 0
            self.backtrack_count += 1
            self.steps.append(('backtrack', row, col, num))
        
        return False
    
    def _is_solvable(self, board: np.ndarray) -> bool:
        """
        Forward checking: Checks if current state can lead to solution
        
        DISCRETE MATH: Pruning impossible branches early
        
        Returns False if any empty cell has no valid options
        """
        for row in range(9):
            for col in range(9):
                if board[row, col] == 0:
                    valid = self.checker.get_valid_numbers(board, row, col)
                    if len(valid) == 0:
                        return False  # Dead end detected
        return True


def solve_sudoku(board: np.ndarray, optimized: bool = True) -> Tuple[bool, np.ndarray, dict]:
    """
    Convenience function to solve Sudoku puzzle
    
    Args:
        board: 9x9 Sudoku board
        optimized: Use optimized solver (default True)
    
    Returns:
        (success, solved_board, statistics) tuple
    """
    if optimized:
        solver = OptimizedBacktrackingSolver()
    else:
        solver = BacktrackingSolver()
    
    success, solved = solver.solve(board)
    stats = solver.get_statistics()
    
    return success, solved, stats


if __name__ == "__main__":
    # Test backtracking solver
    print("Testing Backtracking Solver")
    print("=" * 60)
    
    # Easy puzzle
    easy_puzzle = np.array([
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
    
    print("\nOriginal Puzzle:")
    print(easy_puzzle)
    
    # Solve with basic backtracking
    print("\n--- Basic Backtracking ---")
    solver1 = BacktrackingSolver()
    success1, solved1 = solver1.solve(easy_puzzle.copy(), use_propagation=False)
    stats1 = solver1.get_statistics()
    
    print(f"Solved: {success1}")
    print(f"Time: {stats1['solving_time']:.4f} seconds")
    print(f"Recursions: {stats1['recursion_count']}")
    print(f"Backtracks: {stats1['backtrack_count']}")
    
    # Solve with optimized backtracking
    print("\n--- Optimized Backtracking ---")
    solver2 = OptimizedBacktrackingSolver()
    success2, solved2 = solver2.solve(easy_puzzle.copy())
    stats2 = solver2.get_statistics()
    
    print(f"Solved: {success2}")
    print(f"Time: {stats2['solving_time']:.4f} seconds")
    print(f"Recursions: {stats2['recursion_count']}")
    print(f"Backtracks: {stats2['backtrack_count']}")
    
    print("\nSolved Puzzle:")
    print(solved2)
    
    print("\n" + "=" * 60)
    print("Optimization improved performance by:")
    print(f"Time: {(1 - stats2['solving_time']/stats1['solving_time'])*100:.1f}% faster")
    print(f"Recursions: {(1 - stats2['recursion_count']/stats1['recursion_count'])*100:.1f}% fewer")
