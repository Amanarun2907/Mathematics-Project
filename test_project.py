"""
Quick Test Script for Sudoku Solver Project
===========================================

This script tests all major components to ensure everything works.

Author: Discrete Mathematics Project
"""

import numpy as np
import sys

print("=" * 70)
print("TESTING SUDOKU SOLVER PROJECT")
print("=" * 70)

# Test puzzle
test_puzzle = np.array([
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

print("\n1. Testing Discrete Math Utilities...")
try:
    from utils.discrete_math import SetTheory, PredicateLogic, BooleanAlgebra, Combinatorics, GraphTheory
    
    st = SetTheory()
    assert st.universal_set == set(range(1, 10))
    
    pl = PredicateLogic()
    assert pl.range_constraint(5) == True
    assert pl.range_constraint(10) == False
    
    ba = BooleanAlgebra()
    assert ba.conjunction(True, True) == True
    assert ba.disjunction(False, True) == True
    
    comb = Combinatorics()
    assert comb.factorial(5) == 120
    assert comb.combination(9, 3) == 84
    
    gt = GraphTheory()
    assert gt.chromatic_number() == 9
    
    print("   ✅ Discrete Math Utilities - PASSED")
except Exception as e:
    print(f"   ❌ Discrete Math Utilities - FAILED: {e}")
    sys.exit(1)

print("\n2. Testing Constraint Logic...")
try:
    from solver.constraint_logic import ConstraintChecker, ConstraintPropagation
    
    checker = ConstraintChecker()
    assert checker.is_valid_placement(test_puzzle, 0, 2, 4) == True
    assert checker.is_valid_placement(test_puzzle, 0, 2, 5) == False
    
    valid_nums = checker.get_valid_numbers(test_puzzle, 0, 2)
    assert len(valid_nums) > 0
    
    propagator = ConstraintPropagation()
    test_copy = test_puzzle.copy()
    filled = propagator.propagate_constraints(test_copy)
    assert filled >= 0
    
    print("   ✅ Constraint Logic - PASSED")
except Exception as e:
    print(f"   ❌ Constraint Logic - FAILED: {e}")
    sys.exit(1)

print("\n3. Testing Backtracking Solver...")
try:
    from solver.backtracking_solver import BacktrackingSolver, OptimizedBacktrackingSolver
    
    solver = BacktrackingSolver()
    success, solution = solver.solve(test_puzzle.copy())
    assert success == True
    assert np.sum(solution == 0) == 0  # No empty cells
    
    stats = solver.get_statistics()
    assert stats['recursion_count'] > 0
    assert stats['solving_time'] > 0
    
    print(f"   ✅ Backtracking Solver - PASSED")
    print(f"      Solved in {stats['solving_time']:.4f}s with {stats['recursion_count']} recursions")
except Exception as e:
    print(f"   ❌ Backtracking Solver - FAILED: {e}")
    sys.exit(1)

print("\n4. Testing Solution Validator...")
try:
    from solver.solution_validator import SolutionValidator
    
    validator = SolutionValidator()
    is_valid, errors = validator.validate_solution(solution)
    assert is_valid == True
    assert len(errors) == 0
    
    print("   ✅ Solution Validator - PASSED")
except Exception as e:
    print(f"   ❌ Solution Validator - FAILED: {e}")
    sys.exit(1)

print("\n5. Testing Puzzle Generator...")
try:
    from generator.puzzle_generator import PuzzleGenerator
    
    generator = PuzzleGenerator(seed=42)
    puzzle, solution = generator.generate_puzzle('medium')
    assert puzzle.shape == (9, 9)
    assert solution.shape == (9, 9)
    assert np.sum(puzzle != 0) > 0  # Has given cells
    assert np.sum(solution == 0) == 0  # Solution is complete
    
    print("   ✅ Puzzle Generator - PASSED")
except Exception as e:
    print(f"   ❌ Puzzle Generator - FAILED: {e}")
    sys.exit(1)

print("\n6. Testing Difficulty Analyzer...")
try:
    from generator.difficulty_analyzer import DifficultyAnalyzer, DifficultyEstimator
    
    analyzer = DifficultyAnalyzer()
    analysis = analyzer.analyze_difficulty(test_puzzle)
    assert 'difficulty_score' in analysis
    assert 'difficulty_level' in analysis
    assert analysis['difficulty_level'] in ['Easy', 'Medium', 'Hard', 'Expert']
    
    estimator = DifficultyEstimator()
    score, level = estimator.estimate_difficulty(test_puzzle)
    assert 0 <= score <= 100
    
    print(f"   ✅ Difficulty Analyzer - PASSED")
    print(f"      Puzzle difficulty: {analysis['difficulty_level']} (score: {analysis['difficulty_score']:.2f})")
except Exception as e:
    print(f"   ❌ Difficulty Analyzer - FAILED: {e}")
    sys.exit(1)

print("\n7. Testing ML Models (Structure Only)...")
try:
    from ml_model.neural_solver import NeuralSudokuSolver
    from ml_model.difficulty_classifier import MLDifficultyClassifier
    
    # Just test initialization (not training)
    neural_solver = NeuralSudokuSolver()
    assert neural_solver.model is not None
    
    classifier = MLDifficultyClassifier()
    features = classifier.extract_features(test_puzzle)
    assert len(features) > 0
    
    print("   ✅ ML Models Structure - PASSED")
    print("      Note: Models not trained, only structure tested")
except Exception as e:
    print(f"   ❌ ML Models - FAILED: {e}")
    sys.exit(1)

print("\n8. Testing Visualization...")
try:
    from utils.visualization import SudokuVisualizer, MetricsVisualizer
    
    visualizer = SudokuVisualizer()
    fig = visualizer.create_board_figure(test_puzzle, test_puzzle)
    assert fig is not None
    
    text = visualizer.format_board_text(test_puzzle)
    assert len(text) > 0
    
    print("   ✅ Visualization - PASSED")
except Exception as e:
    print(f"   ❌ Visualization - FAILED: {e}")
    sys.exit(1)

print("\n" + "=" * 70)
print("ALL TESTS PASSED! ✅")
print("=" * 70)
print("\nYour Sudoku Solver project is ready to use!")
print("\nTo run the Streamlit app:")
print("  streamlit run app.py")
print("\n" + "=" * 70)
