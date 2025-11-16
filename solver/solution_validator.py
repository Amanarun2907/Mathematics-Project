"""
Solution Validator for Sudoku
==============================

This module validates Sudoku solutions using discrete mathematics concepts.

DISCRETE MATH CONCEPTS:
1. Predicate Logic - Validation as logical predicates
2. Set Theory - Checking for complete sets
3. Boolean Algebra - Combining validation results

Author: Discrete Mathematics Project
"""

import numpy as np
from typing import List, Tuple, Dict
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from solver.constraint_logic import ConstraintChecker
from utils.discrete_math import SetTheory, PredicateLogic


class SolutionValidator:
    """
    Validates Sudoku solutions comprehensively
    """
    
    def __init__(self):
        self.checker = ConstraintChecker()
        self.set_theory = SetTheory()
        self.predicate_logic = PredicateLogic()
    
    def validate_solution(self, board: np.ndarray) -> Tuple[bool, List[str]]:
        """
        Validates complete Sudoku solution
        
        DISCRETE MATH: Conjunction of all validation predicates
        Valid = Complete ∧ ValidRows ∧ ValidCols ∧ ValidBoxes
        
        Returns:
            (is_valid, error_messages) tuple
        """
        errors = []
        
        # Check if board is complete
        if not self.checker.is_board_complete(board):
            errors.append("Board is not complete (contains empty cells)")
            return False, errors
        
        # Validate all rows
        if not self._validate_rows(board, errors):
            pass  # Errors already added
        
        # Validate all columns
        if not self._validate_columns(board, errors):
            pass  # Errors already added
        
        # Validate all boxes
        if not self._validate_boxes(board, errors):
            pass  # Errors already added
        
        # Check value range
        if not self._validate_range(board, errors):
            pass  # Errors already added
        
        is_valid = len(errors) == 0
        return is_valid, errors
    
    def _validate_rows(self, board: np.ndarray, errors: List[str]) -> bool:
        """
        Validates all rows contain unique numbers 1-9
        
        DISCRETE MATH: Set cardinality check
        For each row: |set(row)| = 9 and set(row) = {1,2,3,4,5,6,7,8,9}
        """
        valid = True
        
        for row in range(9):
            row_set = set(board[row, :])
            
            # Check if row has all 9 unique numbers
            if len(row_set) != 9:
                errors.append(f"Row {row + 1} has duplicate numbers")
                valid = False
            
            # Check if row contains only valid numbers
            if not row_set == self.set_theory.universal_set:
                errors.append(f"Row {row + 1} doesn't contain all numbers 1-9")
                valid = False
        
        return valid
    
    def _validate_columns(self, board: np.ndarray, errors: List[str]) -> bool:
        """
        Validates all columns contain unique numbers 1-9
        
        DISCRETE MATH: Set cardinality check
        For each column: |set(col)| = 9 and set(col) = {1,2,3,4,5,6,7,8,9}
        """
        valid = True
        
        for col in range(9):
            col_set = set(board[:, col])
            
            # Check if column has all 9 unique numbers
            if len(col_set) != 9:
                errors.append(f"Column {col + 1} has duplicate numbers")
                valid = False
            
            # Check if column contains only valid numbers
            if not col_set == self.set_theory.universal_set:
                errors.append(f"Column {col + 1} doesn't contain all numbers 1-9")
                valid = False
        
        return valid
    
    def _validate_boxes(self, board: np.ndarray, errors: List[str]) -> bool:
        """
        Validates all 3x3 boxes contain unique numbers 1-9
        
        DISCRETE MATH: Set cardinality check for subgrids
        For each box: |set(box)| = 9 and set(box) = {1,2,3,4,5,6,7,8,9}
        """
        valid = True
        
        for box_row in range(0, 9, 3):
            for box_col in range(0, 9, 3):
                box = board[box_row:box_row + 3, box_col:box_col + 3]
                box_set = set(box.flatten())
                
                box_num = (box_row // 3) * 3 + (box_col // 3) + 1
                
                # Check if box has all 9 unique numbers
                if len(box_set) != 9:
                    errors.append(f"Box {box_num} has duplicate numbers")
                    valid = False
                
                # Check if box contains only valid numbers
                if not box_set == self.set_theory.universal_set:
                    errors.append(f"Box {box_num} doesn't contain all numbers 1-9")
                    valid = False
        
        return valid
    
    def _validate_range(self, board: np.ndarray, errors: List[str]) -> bool:
        """
        Validates all numbers are in range 1-9
        
        DISCRETE MATH: Domain constraint
        ∀i,j (1 ≤ board[i][j] ≤ 9)
        """
        valid = True
        
        for row in range(9):
            for col in range(9):
                num = board[row, col]
                if not (1 <= num <= 9):
                    errors.append(f"Invalid number {num} at position ({row + 1}, {col + 1})")
                    valid = False
        
        return valid
    
    def validate_partial_solution(self, board: np.ndarray) -> Tuple[bool, List[str]]:
        """
        Validates partial Sudoku solution (with empty cells)
        
        Checks that filled cells don't violate constraints
        
        Returns:
            (is_valid, error_messages) tuple
        """
        errors = []
        
        # Check each filled cell
        for row in range(9):
            for col in range(9):
                if board[row, col] != 0:
                    num = board[row, col]
                    
                    # Temporarily remove to check validity
                    board[row, col] = 0
                    
                    if not self.checker.is_valid_placement(board, row, col, num):
                        errors.append(f"Invalid placement of {num} at ({row + 1}, {col + 1})")
                    
                    # Restore number
                    board[row, col] = num
        
        is_valid = len(errors) == 0
        return is_valid, errors
    
    def get_validation_report(self, board: np.ndarray) -> Dict:
        """
        Generates comprehensive validation report
        
        Returns:
            Dictionary with detailed validation information
        """
        report = {
            'is_complete': self.checker.is_board_complete(board),
            'is_valid': False,
            'errors': [],
            'row_status': [],
            'column_status': [],
            'box_status': [],
            'filled_cells': int(np.sum(board != 0)),
            'empty_cells': int(np.sum(board == 0))
        }
        
        # Validate solution
        is_valid, errors = self.validate_solution(board) if report['is_complete'] else self.validate_partial_solution(board)
        report['is_valid'] = is_valid
        report['errors'] = errors
        
        # Check each row
        for row in range(9):
            row_set = set(board[row, :]) - {0}
            report['row_status'].append({
                'row': row + 1,
                'unique_numbers': len(row_set),
                'is_valid': len(row_set) == len([x for x in board[row, :] if x != 0])
            })
        
        # Check each column
        for col in range(9):
            col_set = set(board[:, col]) - {0}
            report['column_status'].append({
                'column': col + 1,
                'unique_numbers': len(col_set),
                'is_valid': len(col_set) == len([x for x in board[:, col] if x != 0])
            })
        
        # Check each box
        for box_row in range(0, 9, 3):
            for box_col in range(0, 9, 3):
                box = board[box_row:box_row + 3, box_col:box_col + 3]
                box_set = set(box.flatten()) - {0}
                box_num = (box_row // 3) * 3 + (box_col // 3) + 1
                
                report['box_status'].append({
                    'box': box_num,
                    'unique_numbers': len(box_set),
                    'is_valid': len(box_set) == len([x for x in box.flatten() if x != 0])
                })
        
        return report


if __name__ == "__main__":
    # Test validator
    print("Testing Solution Validator")
    print("=" * 60)
    
    # Valid complete solution
    valid_solution = np.array([
        [5, 3, 4, 6, 7, 8, 9, 1, 2],
        [6, 7, 2, 1, 9, 5, 3, 4, 8],
        [1, 9, 8, 3, 4, 2, 5, 6, 7],
        [8, 5, 9, 7, 6, 1, 4, 2, 3],
        [4, 2, 6, 8, 5, 3, 7, 9, 1],
        [7, 1, 3, 9, 2, 4, 8, 5, 6],
        [9, 6, 1, 5, 3, 7, 2, 8, 4],
        [2, 8, 7, 4, 1, 9, 6, 3, 5],
        [3, 4, 5, 2, 8, 6, 1, 7, 9]
    ])
    
    validator = SolutionValidator()
    
    print("\nValidating correct solution:")
    is_valid, errors = validator.validate_solution(valid_solution)
    print(f"Valid: {is_valid}")
    if errors:
        for error in errors:
            print(f"  - {error}")
    
    # Invalid solution (duplicate in row)
    invalid_solution = valid_solution.copy()
    invalid_solution[0, 0] = 3  # Duplicate 3 in row 0
    
    print("\nValidating invalid solution (duplicate in row):")
    is_valid, errors = validator.validate_solution(invalid_solution)
    print(f"Valid: {is_valid}")
    if errors:
        for error in errors:
            print(f"  - {error}")
    
    # Partial solution
    partial = valid_solution.copy()
    partial[0, 0] = 0
    partial[1, 1] = 0
    
    print("\nValidating partial solution:")
    is_valid, errors = validator.validate_partial_solution(partial)
    print(f"Valid: {is_valid}")
    
    # Get detailed report
    print("\nDetailed validation report:")
    report = validator.get_validation_report(partial)
    print(f"Complete: {report['is_complete']}")
    print(f"Valid: {report['is_valid']}")
    print(f"Filled cells: {report['filled_cells']}")
    print(f"Empty cells: {report['empty_cells']}")
