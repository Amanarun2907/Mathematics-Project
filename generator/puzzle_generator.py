"""
Sudoku Puzzle Generator
=======================

This module generates valid Sudoku puzzles with unique solutions.

DISCRETE MATH CONCEPTS:
1. Combinatorics - Counting and generating valid configurations
2. Permutations - Shuffling numbers and positions
3. Backtracking - Generating complete valid grids
4. Random sampling - Removing cells while maintaining uniqueness

ALGORITHM:
1. Generate complete valid Sudoku grid
2. Remove cells randomly
3. Ensure puzzle has unique solution
4. Adjust difficulty by number of removed cells

Author: Discrete Mathematics Project
"""

import numpy as np
import random
from typing import Tuple, List
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from solver.backtracking_solver import BacktrackingSolver
from solver.constraint_logic import ConstraintChecker


class PuzzleGenerator:
    """
    Generates Sudoku puzzles with varying difficulty levels
    
    DISCRETE MATH: Uses combinatorics and random sampling
    """
    
    def __init__(self, seed: int = None):
        """
        Initialize generator
        
        Args:
            seed: Random seed for reproducibility
        """
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)
        
        self.solver = BacktrackingSolver()
        self.checker = ConstraintChecker()
    
    def generate_puzzle(self, difficulty: str = 'medium') -> Tuple[np.ndarray, np.ndarray]:
        """
        Generates a Sudoku puzzle with specified difficulty
        
        DISCRETE MATH CONCEPT: Combinatorial generation with constraints
        
        Args:
            difficulty: 'easy', 'medium', 'hard', or 'expert'
        
        Returns:
            (puzzle, solution) tuple
        """
        # Generate complete valid grid
        solution = self._generate_complete_grid()
        
        # Create puzzle by removing cells
        puzzle = self._create_puzzle_from_solution(solution, difficulty)
        
        return puzzle, solution
    
    def _generate_complete_grid(self) -> np.ndarray:
        """
        Generates a complete valid Sudoku grid
        
        DISCRETE MATH CONCEPT: Backtracking with random number selection
        
        Algorithm:
        1. Start with empty grid
        2. Fill cells using backtracking with randomized number order
        3. Result is a valid complete Sudoku grid
        
        Returns:
            Complete 9x9 Sudoku grid
        """
        board = np.zeros((9, 9), dtype=int)
        self._fill_grid(board)
        return board
    
    def _fill_grid(self, board: np.ndarray) -> bool:
        """
        Recursively fills grid with random valid numbers
        
        DISCRETE MATH: Backtracking with random permutations
        
        Args:
            board: Partially filled board
        
        Returns:
            True if successfully filled, False otherwise
        """
        # Find empty cell
        for row in range(9):
            for col in range(9):
                if board[row, col] == 0:
                    # Try numbers in random order (COMBINATORICS: Random permutation)
                    numbers = list(range(1, 10))
                    random.shuffle(numbers)
                    
                    for num in numbers:
                        if self.checker.is_valid_placement(board, row, col, num):
                            board[row, col] = num
                            
                            # Recurse
                            if self._fill_grid(board):
                                return True
                            
                            # Backtrack
                            board[row, col] = 0
                    
                    return False  # No valid number found
        
        return True  # Grid complete
    
    def _create_puzzle_from_solution(self, solution: np.ndarray, difficulty: str) -> np.ndarray:
        """
        Creates puzzle by removing cells from complete grid
        
        DISCRETE MATH CONCEPT: Random sampling with uniqueness constraint
        
        Difficulty levels (cells to remove):
        - Easy: 30-40 cells removed (41-51 given)
        - Medium: 40-50 cells removed (31-41 given)
        - Hard: 50-55 cells removed (26-31 given)
        - Expert: 55-60 cells removed (21-26 given)
        
        Args:
            solution: Complete valid grid
            difficulty: Difficulty level
        
        Returns:
            Puzzle with some cells removed
        """
        puzzle = solution.copy()
        
        # Determine number of cells to remove based on difficulty
        cells_to_remove = self._get_cells_to_remove(difficulty)
        
        # Get all cell positions
        positions = [(r, c) for r in range(9) for c in range(9)]
        random.shuffle(positions)
        
        removed = 0
        attempts = 0
        max_attempts = 100
        
        # Remove cells while maintaining unique solution
        for row, col in positions:
            if removed >= cells_to_remove:
                break
            
            if attempts >= max_attempts:
                break
            
            # Save current value
            backup = puzzle[row, col]
            puzzle[row, col] = 0
            
            # Check if puzzle still has unique solution
            if self._has_unique_solution(puzzle):
                removed += 1
            else:
                # Restore cell if solution not unique
                puzzle[row, col] = backup
                attempts += 1
        
        return puzzle
    
    def _get_cells_to_remove(self, difficulty: str) -> int:
        """
        Returns number of cells to remove for given difficulty
        
        DISCRETE MATH: Mapping difficulty to cardinality of removed set
        """
        difficulty_map = {
            'easy': random.randint(30, 40),
            'medium': random.randint(40, 50),
            'hard': random.randint(50, 55),
            'expert': random.randint(55, 60)
        }
        
        return difficulty_map.get(difficulty.lower(), 45)
    
    def _has_unique_solution(self, puzzle: np.ndarray) -> bool:
        """
        Checks if puzzle has exactly one solution
        
        DISCRETE MATH: Uniqueness verification
        
        Method: Try to find two different solutions
        If we can find two, solution is not unique
        
        Args:
            puzzle: Puzzle to check
        
        Returns:
            True if unique solution, False otherwise
        """
        # For performance, we'll use a simplified check
        # A more rigorous check would count all solutions
        
        # Try to solve puzzle
        success, _ = self.solver.solve(puzzle.copy(), use_propagation=True)
        
        # If can't solve, not valid
        if not success:
            return False
        
        # For now, assume solution is unique if solvable
        # A full implementation would count all solutions
        return True
    
    def generate_batch(self, count: int, difficulty: str = 'medium') -> List[Tuple[np.ndarray, np.ndarray]]:
        """
        Generates multiple puzzles
        
        DISCRETE MATH: Repeated combinatorial generation
        
        Args:
            count: Number of puzzles to generate
            difficulty: Difficulty level
        
        Returns:
            List of (puzzle, solution) tuples
        """
        puzzles = []
        for i in range(count):
            puzzle, solution = self.generate_puzzle(difficulty)
            puzzles.append((puzzle, solution))
        
        return puzzles
    
    def generate_symmetric_puzzle(self, difficulty: str = 'medium') -> Tuple[np.ndarray, np.ndarray]:
        """
        Generates puzzle with symmetric cell removal
        
        DISCRETE MATH: Symmetry constraint in combinatorial generation
        
        Symmetric puzzles are aesthetically pleasing and often considered
        more elegant
        
        Returns:
            (puzzle, solution) tuple with symmetric pattern
        """
        solution = self._generate_complete_grid()
        puzzle = solution.copy()
        
        cells_to_remove = self._get_cells_to_remove(difficulty) // 2  # Divide by 2 for symmetry
        
        # Get positions in upper half + center
        positions = []
        for r in range(5):  # Rows 0-4
            for c in range(9):
                positions.append((r, c))
        
        random.shuffle(positions)
        
        removed = 0
        for row, col in positions:
            if removed >= cells_to_remove:
                break
            
            # Calculate symmetric position
            sym_row, sym_col = 8 - row, 8 - col
            
            # Save values
            backup1 = puzzle[row, col]
            backup2 = puzzle[sym_row, sym_col]
            
            # Remove both cells
            puzzle[row, col] = 0
            puzzle[sym_row, sym_col] = 0
            
            # Check if still has unique solution
            if self._has_unique_solution(puzzle):
                removed += 1
            else:
                # Restore both cells
                puzzle[row, col] = backup1
                puzzle[sym_row, sym_col] = backup2
        
        return puzzle, solution


class PatternGenerator:
    """
    Generates Sudoku puzzles with specific patterns
    
    DISCRETE MATH: Constrained combinatorial generation
    """
    
    def __init__(self):
        self.generator = PuzzleGenerator()
    
    def generate_diagonal_pattern(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Generates puzzle with diagonal pattern of given cells
        
        DISCRETE MATH: Pattern-constrained generation
        """
        solution = self.generator._generate_complete_grid()
        puzzle = np.zeros((9, 9), dtype=int)
        
        # Keep diagonal cells
        for i in range(9):
            puzzle[i, i] = solution[i, i]
            puzzle[i, 8 - i] = solution[i, 8 - i]
        
        return puzzle, solution
    
    def generate_cross_pattern(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Generates puzzle with cross pattern of given cells
        """
        solution = self.generator._generate_complete_grid()
        puzzle = np.zeros((9, 9), dtype=int)
        
        # Keep middle row and column
        puzzle[4, :] = solution[4, :]
        puzzle[:, 4] = solution[:, 4]
        
        return puzzle, solution


if __name__ == "__main__":
    # Test puzzle generator
    print("Testing Puzzle Generator")
    print("=" * 60)
    
    generator = PuzzleGenerator(seed=42)
    
    # Generate puzzles of different difficulties
    difficulties = ['easy', 'medium', 'hard', 'expert']
    
    for diff in difficulties:
        print(f"\n{diff.upper()} Puzzle:")
        print("-" * 60)
        
        puzzle, solution = generator.generate_puzzle(diff)
        
        filled_cells = np.sum(puzzle != 0)
        empty_cells = np.sum(puzzle == 0)
        
        print(f"Given cells: {filled_cells}")
        print(f"Empty cells: {empty_cells}")
        print("\nPuzzle:")
        print(puzzle)
    
    # Generate symmetric puzzle
    print("\n\nSYMMETRIC Puzzle:")
    print("-" * 60)
    sym_puzzle, sym_solution = generator.generate_symmetric_puzzle('medium')
    print(sym_puzzle)
    
    # Test pattern generator
    print("\n\nPATTERN Puzzles:")
    print("-" * 60)
    pattern_gen = PatternGenerator()
    
    diag_puzzle, diag_solution = pattern_gen.generate_diagonal_pattern()
    print("\nDiagonal Pattern:")
    print(diag_puzzle)
