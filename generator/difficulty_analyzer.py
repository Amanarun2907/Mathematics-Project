"""
Difficulty Analyzer for Sudoku Puzzles
======================================

This module analyzes and classifies Sudoku puzzle difficulty.

DISCRETE MATH CONCEPTS:
1. Graph Theory - Constraint graph analysis
2. Combinatorics - Counting possibilities
3. Set Theory - Analyzing constraint sets
4. Heuristics - Difficulty estimation functions

DIFFICULTY METRICS:
1. Number of given cells
2. Number of naked singles
3. Number of hidden singles
4. Constraint density
5. Branching factor
6. Backtracking depth required

Author: Discrete Mathematics Project
"""

import numpy as np
from typing import Dict, Tuple
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from solver.backtracking_solver import BacktrackingSolver
from solver.constraint_logic import ConstraintChecker, ConstraintPropagation
from utils.discrete_math import Combinatorics, GraphTheory


class DifficultyAnalyzer:
    """
    Analyzes Sudoku puzzle difficulty using multiple metrics
    
    DISCRETE MATH: Multi-criteria analysis using various concepts
    """
    
    def __init__(self):
        self.solver = BacktrackingSolver()
        self.checker = ConstraintChecker()
        self.propagator = ConstraintPropagation()
        self.combinatorics = Combinatorics()
        self.graph_theory = GraphTheory()
    
    def analyze_difficulty(self, puzzle: np.ndarray) -> Dict:
        """
        Comprehensive difficulty analysis
        
        Returns dictionary with multiple difficulty metrics
        
        Args:
            puzzle: Sudoku puzzle to analyze
        
        Returns:
            Dictionary with difficulty metrics and classification
        """
        metrics = {}
        
        # Basic metrics
        metrics['given_cells'] = int(np.sum(puzzle != 0))
        metrics['empty_cells'] = int(np.sum(puzzle == 0))
        metrics['fill_percentage'] = (metrics['given_cells'] / 81) * 100
        
        # Constraint analysis
        metrics.update(self._analyze_constraints(puzzle))
        
        # Solving complexity
        metrics.update(self._analyze_solving_complexity(puzzle))
        
        # Technique requirements
        metrics.update(self._analyze_techniques(puzzle))
        
        # Calculate overall difficulty score
        metrics['difficulty_score'] = self._calculate_difficulty_score(metrics)
        metrics['difficulty_level'] = self._classify_difficulty(metrics['difficulty_score'])
        
        return metrics
    
    def _analyze_constraints(self, puzzle: np.ndarray) -> Dict:
        """
        Analyzes constraint structure
        
        DISCRETE MATH: Set theory and graph theory analysis
        """
        metrics = {}
        
        # Count constraints for each empty cell
        constraint_counts = []
        option_counts = []
        
        for row in range(9):
            for col in range(9):
                if puzzle[row, col] == 0:
                    # Number of constraints (used numbers affecting this cell)
                    constraints = self.checker.count_constraints(puzzle, row, col)
                    constraint_counts.append(constraints)
                    
                    # Number of valid options
                    options = len(self.checker.get_valid_numbers(puzzle, row, col))
                    option_counts.append(options)
        
        if constraint_counts:
            metrics['avg_constraints'] = np.mean(constraint_counts)
            metrics['max_constraints'] = np.max(constraint_counts)
            metrics['min_constraints'] = np.min(constraint_counts)
        else:
            metrics['avg_constraints'] = 0
            metrics['max_constraints'] = 0
            metrics['min_constraints'] = 0
        
        if option_counts:
            metrics['avg_options'] = np.mean(option_counts)
            metrics['max_options'] = np.max(option_counts)
            metrics['min_options'] = np.min(option_counts)
            metrics['cells_with_few_options'] = sum(1 for x in option_counts if x <= 2)
        else:
            metrics['avg_options'] = 0
            metrics['max_options'] = 0
            metrics['min_options'] = 0
            metrics['cells_with_few_options'] = 0
        
        return metrics
    
    def _analyze_solving_complexity(self, puzzle: np.ndarray) -> Dict:
        """
        Analyzes solving complexity by actually solving
        
        DISCRETE MATH: Backtracking complexity analysis
        """
        metrics = {}
        
        # Solve puzzle and get statistics
        puzzle_copy = puzzle.copy()
        success, solved = self.solver.solve(puzzle_copy)
        
        if success:
            stats = self.solver.get_statistics()
            metrics['recursion_depth'] = stats['recursion_count']
            metrics['backtrack_count'] = stats['backtrack_count']
            metrics['solving_time'] = stats['solving_time']
            metrics['solvable'] = True
        else:
            metrics['recursion_depth'] = 0
            metrics['backtrack_count'] = 0
            metrics['solving_time'] = 0
            metrics['solvable'] = False
        
        return metrics
    
    def _analyze_techniques(self, puzzle: np.ndarray) -> Dict:
        """
        Analyzes which solving techniques are required
        
        DISCRETE MATH: Constraint propagation analysis
        """
        metrics = {}
        
        # Get empty cells count
        empty_cells = int(np.sum(puzzle == 0))
        
        puzzle_copy = puzzle.copy()
        
        # Count naked singles
        ns_changed, ns_count = self.propagator.naked_singles(puzzle_copy)
        metrics['naked_singles_count'] = ns_count
        
        # Count hidden singles
        puzzle_copy = puzzle.copy()
        hs_changed, hs_count = self.propagator.hidden_singles(puzzle_copy)
        metrics['hidden_singles_count'] = hs_count
        
        # Total cells solvable by constraint propagation
        puzzle_copy = puzzle.copy()
        total_propagated = self.propagator.propagate_constraints(puzzle_copy)
        metrics['constraint_propagation_fills'] = total_propagated
        
        # Cells requiring backtracking
        metrics['backtracking_required'] = empty_cells - total_propagated
        
        return metrics
    
    def _calculate_difficulty_score(self, metrics: Dict) -> float:
        """
        Calculates overall difficulty score (0-100)
        
        DISCRETE MATH: Weighted combination of metrics
        
        Higher score = more difficult
        """
        score = 0.0
        
        # Empty cells contribute to difficulty (0-30 points)
        empty_ratio = metrics.get('empty_cells', 0) / 81
        score += empty_ratio * 30
        
        # Average options (fewer options = harder) (0-20 points)
        if metrics.get('avg_options', 0) > 0:
            options_score = (9 - metrics['avg_options']) / 8 * 20
            score += options_score
        
        # Backtracking requirement (0-25 points)
        if metrics.get('empty_cells', 0) > 0:
            backtrack_ratio = metrics.get('backtracking_required', 0) / metrics['empty_cells']
            score += backtrack_ratio * 25
        
        # Recursion depth (0-15 points)
        if metrics.get('recursion_depth', 0) > 0 and metrics.get('empty_cells', 0) > 0:
            # Normalize by empty cells
            recursion_score = min(metrics['recursion_depth'] / (metrics['empty_cells'] * 10), 1.0) * 15
            score += recursion_score
        
        # Cells with few options (0-10 points)
        if metrics.get('empty_cells', 0) > 0:
            few_options_ratio = metrics.get('cells_with_few_options', 0) / metrics['empty_cells']
            score += (1 - few_options_ratio) * 10
        
        return min(score, 100)  # Cap at 100
    
    def _classify_difficulty(self, score: float) -> str:
        """
        Classifies difficulty based on score
        
        DISCRETE MATH: Discrete classification function
        """
        if score < 25:
            return 'Easy'
        elif score < 50:
            return 'Medium'
        elif score < 75:
            return 'Hard'
        else:
            return 'Expert'
    
    def compare_puzzles(self, puzzles: list) -> Dict:
        """
        Compares difficulty of multiple puzzles
        
        Args:
            puzzles: List of Sudoku puzzles
        
        Returns:
            Comparison statistics
        """
        analyses = [self.analyze_difficulty(p) for p in puzzles]
        
        scores = [a['difficulty_score'] for a in analyses]
        levels = [a['difficulty_level'] for a in analyses]
        
        comparison = {
            'count': len(puzzles),
            'avg_score': np.mean(scores),
            'min_score': np.min(scores),
            'max_score': np.max(scores),
            'std_score': np.std(scores),
            'level_distribution': {
                'Easy': levels.count('Easy'),
                'Medium': levels.count('Medium'),
                'Hard': levels.count('Hard'),
                'Expert': levels.count('Expert')
            }
        }
        
        return comparison
    
    def get_difficulty_breakdown(self, puzzle: np.ndarray) -> str:
        """
        Returns human-readable difficulty breakdown
        
        Args:
            puzzle: Sudoku puzzle
        
        Returns:
            Formatted string with difficulty analysis
        """
        metrics = self.analyze_difficulty(puzzle)
        
        breakdown = f"""
SUDOKU DIFFICULTY ANALYSIS
{'=' * 60}

BASIC METRICS:
  Given Cells: {metrics['given_cells']} ({metrics['fill_percentage']:.1f}%)
  Empty Cells: {metrics['empty_cells']}

CONSTRAINT ANALYSIS:
  Average Constraints per Cell: {metrics['avg_constraints']:.2f}
  Average Options per Cell: {metrics['avg_options']:.2f}
  Cells with Few Options (≤2): {metrics['cells_with_few_options']}

SOLVING COMPLEXITY:
  Recursion Depth: {metrics['recursion_depth']}
  Backtrack Count: {metrics['backtrack_count']}
  Solving Time: {metrics['solving_time']:.4f} seconds

TECHNIQUE REQUIREMENTS:
  Naked Singles: {metrics['naked_singles_count']}
  Hidden Singles: {metrics['hidden_singles_count']}
  Constraint Propagation Fills: {metrics['constraint_propagation_fills']}
  Cells Requiring Backtracking: {metrics['backtracking_required']}

OVERALL ASSESSMENT:
  Difficulty Score: {metrics['difficulty_score']:.2f} / 100
  Difficulty Level: {metrics['difficulty_level']}
{'=' * 60}
"""
        return breakdown


class DifficultyEstimator:
    """
    Quick difficulty estimation without solving
    
    DISCRETE MATH: Heuristic estimation using combinatorics
    """
    
    def __init__(self):
        self.checker = ConstraintChecker()
    
    def estimate_difficulty(self, puzzle: np.ndarray) -> Tuple[float, str]:
        """
        Quickly estimates difficulty without full solving
        
        Uses heuristics based on:
        1. Number of given cells
        2. Distribution of constraints
        3. Minimum options available
        
        Returns:
            (score, level) tuple
        """
        given = np.sum(puzzle != 0)
        empty = np.sum(puzzle == 0)
        
        # Count cells by number of options
        option_distribution = {i: 0 for i in range(1, 10)}
        
        for row in range(9):
            for col in range(9):
                if puzzle[row, col] == 0:
                    options = len(self.checker.get_valid_numbers(puzzle, row, col))
                    if options > 0:
                        option_distribution[options] += 1
        
        # Calculate heuristic score
        score = 0.0
        
        # Fewer given cells = harder
        score += (empty / 81) * 40
        
        # Cells with many options = harder
        for num_options, count in option_distribution.items():
            if count > 0:
                score += (num_options / 9) * (count / empty) * 30
        
        # Cells with very few options = easier (can be filled quickly)
        if option_distribution[1] + option_distribution[2] > 0:
            easy_cells_ratio = (option_distribution[1] + option_distribution[2]) / empty
            score -= easy_cells_ratio * 20
        
        score = max(0, min(100, score))
        
        # Classify
        if score < 25:
            level = 'Easy'
        elif score < 50:
            level = 'Medium'
        elif score < 75:
            level = 'Hard'
        else:
            level = 'Expert'
        
        return score, level


if __name__ == "__main__":
    # Test difficulty analyzer
    print("Testing Difficulty Analyzer")
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
    
    analyzer = DifficultyAnalyzer()
    
    print("\nAnalyzing puzzle...")
    breakdown = analyzer.get_difficulty_breakdown(easy_puzzle)
    print(breakdown)
    
    # Quick estimation
    print("\nQuick Estimation:")
    estimator = DifficultyEstimator()
    score, level = estimator.estimate_difficulty(easy_puzzle)
    print(f"Estimated Score: {score:.2f}")
    print(f"Estimated Level: {level}")
