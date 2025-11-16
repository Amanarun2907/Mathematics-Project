"""
Constraint Logic and Predicate Logic Implementation
===================================================

This module implements Sudoku constraints using predicate logic from discrete mathematics.

DISCRETE MATH CONCEPTS:
1. Predicate Logic - Constraints as logical predicates
2. Boolean Algebra - Logical operations (AND, OR, NOT)
3. Set Theory - Valid number sets and constraint sets

SUDOKU CONSTRAINTS AS PREDICATES:
- Row Constraint: ∀i,j,k (i≠k → cell[i][j] ≠ cell[k][j])
- Column Constraint: ∀i,j,k (j≠k → cell[i][j] ≠ cell[i][k])
- Box Constraint: ∀cells in 3x3 box (all different)
- Domain Constraint: ∀i,j (cell[i][j] ∈ {1,2,3,4,5,6,7,8,9} ∪ {0})

Author: Discrete Mathematics Project
"""

import numpy as np
from typing import Set, List, Tuple, Dict
import sys
import os

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.discrete_math import SetTheory, PredicateLogic, BooleanAlgebra


class ConstraintChecker:
    """
    Implements constraint checking using predicate logic
    
    DISCRETE MATH: Each constraint is a predicate (boolean-valued function)
    that returns True if satisfied, False otherwise
    """
    
    def __init__(self):
        self.set_theory = SetTheory()
        self.predicate_logic = PredicateLogic()
        self.boolean_algebra = BooleanAlgebra()
    
    def is_valid_placement(self, board: np.ndarray, row: int, col: int, num: int) -> bool:
        """
        Checks if placing 'num' at (row, col) satisfies ALL constraints
        
        DISCRETE MATH CONCEPT: Conjunction of predicates
        Valid = P_row ∧ P_col ∧ P_box ∧ P_domain
        
        Args:
            board: 9x9 Sudoku board
            row: Row index (0-8)
            col: Column index (0-8)
            num: Number to place (1-9)
        
        Returns:
            True if placement is valid, False otherwise
        """
        # Use predicate logic to check all constraints
        return self.predicate_logic.all_constraints(board, row, col, num)
    
    def get_valid_numbers(self, board: np.ndarray, row: int, col: int) -> Set[int]:
        """
        Returns set of valid numbers for a cell using set theory
        
        DISCRETE MATH CONCEPT: Set difference
        Valid = Universal_Set - (Row_Used ∪ Col_Used ∪ Box_Used)
        
        Args:
            board: 9x9 Sudoku board
            row: Row index
            col: Column index
        
        Returns:
            Set of valid numbers (1-9) that can be placed
        """
        # Get used numbers from row, column, and box
        row_used = self._get_row_numbers(board, row)
        col_used = self._get_column_numbers(board, col)
        box_used = self._get_box_numbers(board, row, col)
        
        # Union of all used numbers (Set Theory: A ∪ B ∪ C)
        all_used = self.set_theory.union_sets(row_used, col_used, box_used)
        
        # Get available numbers (Set Theory: U - A)
        available = self.set_theory.get_available_numbers(all_used)
        
        return available
    
    def _get_row_numbers(self, board: np.ndarray, row: int) -> Set[int]:
        """
        Returns set of numbers already in the row
        
        DISCRETE MATH: Extracts elements from row into a set
        """
        return set(board[row, :]) - {0}  # Exclude empty cells (0)
    
    def _get_column_numbers(self, board: np.ndarray, col: int) -> Set[int]:
        """
        Returns set of numbers already in the column
        
        DISCRETE MATH: Extracts elements from column into a set
        """
        return set(board[:, col]) - {0}  # Exclude empty cells (0)
    
    def _get_box_numbers(self, board: np.ndarray, row: int, col: int) -> Set[int]:
        """
        Returns set of numbers already in the 3x3 box
        
        DISCRETE MATH: Extracts elements from 3x3 subgrid into a set
        Box position determined by integer division: (row//3, col//3)
        """
        box_row, box_col = 3 * (row // 3), 3 * (col // 3)
        box = board[box_row:box_row + 3, box_col:box_col + 3]
        return set(box.flatten()) - {0}  # Exclude empty cells (0)
    
    def count_constraints(self, board: np.ndarray, row: int, col: int) -> int:
        """
        Counts number of constraints (used numbers) affecting a cell
        
        DISCRETE MATH: Cardinality of constraint set
        |Row_Used ∪ Col_Used ∪ Box_Used|
        
        Higher count = more constrained = harder to fill
        """
        row_used = self._get_row_numbers(board, row)
        col_used = self._get_column_numbers(board, col)
        box_used = self._get_box_numbers(board, row, col)
        
        all_used = self.set_theory.union_sets(row_used, col_used, box_used)
        return self.set_theory.cardinality(all_used)
    
    def is_board_valid(self, board: np.ndarray) -> bool:
        """
        Checks if entire board satisfies all constraints
        
        DISCRETE MATH: Universal quantification
        ∀i,j (cell[i][j] = 0 ∨ valid_placement(i, j, cell[i][j]))
        
        Returns:
            True if board is valid (may be incomplete), False otherwise
        """
        for row in range(9):
            for col in range(9):
                if board[row, col] != 0:
                    num = board[row, col]
                    # Temporarily remove number to check if it's valid
                    board[row, col] = 0
                    if not self.is_valid_placement(board, row, col, num):
                        board[row, col] = num
                        return False
                    board[row, col] = num
        return True
    
    def is_board_complete(self, board: np.ndarray) -> bool:
        """
        Checks if board is completely filled
        
        DISCRETE MATH: Existential quantification
        ¬∃i,j (cell[i][j] = 0)
        Equivalent to: ∀i,j (cell[i][j] ≠ 0)
        
        Returns:
            True if no empty cells, False otherwise
        """
        return not np.any(board == 0)
    
    def is_solved(self, board: np.ndarray) -> bool:
        """
        Checks if board is completely solved
        
        DISCRETE MATH: Conjunction of predicates
        Solved = Complete ∧ Valid
        
        Returns:
            True if board is complete and valid, False otherwise
        """
        return self.boolean_algebra.conjunction(
            self.is_board_complete(board),
            self.is_board_valid(board)
        )


class ConstraintPropagation:
    """
    Implements constraint propagation techniques
    
    DISCRETE MATH CONCEPT: Iterative constraint satisfaction
    Reduces search space by eliminating impossible values
    """
    
    def __init__(self):
        self.checker = ConstraintChecker()
        self.set_theory = SetTheory()
    
    def naked_singles(self, board: np.ndarray) -> Tuple[bool, int]:
        """
        Finds cells with only one possible value (naked singles)
        
        DISCRETE MATH: Singleton set detection
        If |Valid_Numbers(cell)| = 1, then cell must have that value
        
        Returns:
            (changed, count) - whether board changed and number of fills
        """
        changed = False
        count = 0
        
        for row in range(9):
            for col in range(9):
                if board[row, col] == 0:
                    valid = self.checker.get_valid_numbers(board, row, col)
                    
                    # If only one valid number (singleton set)
                    if len(valid) == 1:
                        num = list(valid)[0]
                        board[row, col] = num
                        changed = True
                        count += 1
        
        return changed, count
    
    def hidden_singles(self, board: np.ndarray) -> Tuple[bool, int]:
        """
        Finds numbers that can only go in one cell in a unit
        
        DISCRETE MATH: Uniqueness constraint
        If number n can only be placed in one cell in row/col/box,
        then it must go there
        
        Returns:
            (changed, count) - whether board changed and number of fills
        """
        changed = False
        count = 0
        
        # Check rows
        for row in range(9):
            for num in range(1, 10):
                possible_cols = []
                for col in range(9):
                    if board[row, col] == 0 and self.checker.is_valid_placement(board, row, col, num):
                        possible_cols.append(col)
                
                if len(possible_cols) == 1:
                    col = possible_cols[0]
                    board[row, col] = num
                    changed = True
                    count += 1
        
        # Check columns
        for col in range(9):
            for num in range(1, 10):
                possible_rows = []
                for row in range(9):
                    if board[row, col] == 0 and self.checker.is_valid_placement(board, row, col, num):
                        possible_rows.append(row)
                
                if len(possible_rows) == 1:
                    row = possible_rows[0]
                    board[row, col] = num
                    changed = True
                    count += 1
        
        # Check boxes
        for box_row in range(0, 9, 3):
            for box_col in range(0, 9, 3):
                for num in range(1, 10):
                    possible_cells = []
                    for r in range(box_row, box_row + 3):
                        for c in range(box_col, box_col + 3):
                            if board[r, c] == 0 and self.checker.is_valid_placement(board, r, c, num):
                                possible_cells.append((r, c))
                    
                    if len(possible_cells) == 1:
                        r, c = possible_cells[0]
                        board[r, c] = num
                        changed = True
                        count += 1
        
        return changed, count
    
    def propagate_constraints(self, board: np.ndarray, max_iterations: int = 10) -> int:
        """
        Applies constraint propagation iteratively
        
        DISCRETE MATH: Fixed-point iteration
        Repeat until no changes occur (fixed point reached)
        
        Args:
            board: Sudoku board (modified in place)
            max_iterations: Maximum iterations to prevent infinite loops
        
        Returns:
            Total number of cells filled
        """
        total_filled = 0
        
        for iteration in range(max_iterations):
            changed = False
            
            # Apply naked singles
            ns_changed, ns_count = self.naked_singles(board)
            changed = changed or ns_changed
            total_filled += ns_count
            
            # Apply hidden singles
            hs_changed, hs_count = self.hidden_singles(board)
            changed = changed or hs_changed
            total_filled += hs_count
            
            # If no changes, we've reached a fixed point
            if not changed:
                break
        
        return total_filled


class ConstraintAnalyzer:
    """
    Analyzes constraint structure of Sudoku puzzles
    
    Used for difficulty estimation and puzzle generation
    """
    
    def __init__(self):
        self.checker = ConstraintChecker()
    
    def analyze_puzzle(self, board: np.ndarray) -> Dict[str, any]:
        """
        Analyzes puzzle constraints and returns metrics
        
        Returns:
            Dictionary with analysis results
        """
        analysis = {
            'empty_cells': int(np.sum(board == 0)),
            'filled_cells': int(np.sum(board != 0)),
            'constraint_density': {},
            'min_options': 9,
            'max_options': 0,
            'avg_options': 0.0
        }
        
        total_options = 0
        option_count = 0
        
        for row in range(9):
            for col in range(9):
                if board[row, col] == 0:
                    valid = self.checker.get_valid_numbers(board, row, col)
                    num_options = len(valid)
                    
                    analysis['constraint_density'][(row, col)] = num_options
                    analysis['min_options'] = min(analysis['min_options'], num_options)
                    analysis['max_options'] = max(analysis['max_options'], num_options)
                    
                    total_options += num_options
                    option_count += 1
        
        if option_count > 0:
            analysis['avg_options'] = total_options / option_count
        
        return analysis


if __name__ == "__main__":
    # Test constraint logic
    print("Testing Constraint Logic System")
    print("=" * 60)
    
    # Create a sample board
    board = np.array([
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
    
    checker = ConstraintChecker()
    
    print("\nChecking valid placements:")
    print(f"Can place 4 at (0,2)? {checker.is_valid_placement(board, 0, 2, 4)}")
    print(f"Can place 5 at (0,2)? {checker.is_valid_placement(board, 0, 2, 5)}")
    
    print(f"\nValid numbers for (0,2): {checker.get_valid_numbers(board, 0, 2)}")
    print(f"Constraint count for (0,2): {checker.count_constraints(board, 0, 2)}")
    
    print(f"\nIs board valid? {checker.is_board_valid(board)}")
    print(f"Is board complete? {checker.is_board_complete(board)}")
    print(f"Is board solved? {checker.is_solved(board)}")
    
    print("\nApplying constraint propagation...")
    propagator = ConstraintPropagation()
    filled = propagator.propagate_constraints(board.copy())
    print(f"Filled {filled} cells using constraint propagation")
