"""
Sudoku Solver - Streamlit Application
=====================================

Main application interface for Sudoku solving with discrete mathematics concepts.

FEATURES:
1. Interactive puzzle input
2. Multiple solving methods (Backtracking, Neural Network)
3. Difficulty analysis and classification
4. Step-by-step visualization
5. Puzzle generation
6. Educational discrete math explanations

Author: Discrete Mathematics Project
"""

import streamlit as st
import numpy as np
import pandas as pd
import time
from typing import Tuple

# Import solver modules
from solver.backtracking_solver import BacktrackingSolver, OptimizedBacktrackingSolver
from solver.constraint_logic import ConstraintChecker, ConstraintPropagation
from solver.solution_validator import SolutionValidator

# Import generator modules
from generator.puzzle_generator import PuzzleGenerator, PatternGenerator
from generator.difficulty_analyzer import DifficultyAnalyzer, DifficultyEstimator

# Import ML modules
from ml_model.neural_solver import NeuralSudokuSolver
from ml_model.difficulty_classifier import MLDifficultyClassifier, HybridDifficultyClassifier

# Import utilities
from utils.discrete_math import SetTheory, PredicateLogic, BooleanAlgebra, Combinatorics, GraphTheory
from utils.visualization import SudokuVisualizer, MetricsVisualizer

# Page configuration
st.set_page_config(
    page_title="Sudoku Solver - Discrete Mathematics",
    page_icon="🧮",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        text-align: center;
        color: #1f77b4;
        margin-bottom: 1rem;
    }
    .sub-header {
        font-size: 1.5rem;
        color: #ff7f0e;
        margin-top: 1rem;
    }
</style>
""", unsafe_allow_html=True)


# Initialize session state
def init_session_state():
    """Initialize session state variables"""
    if 'puzzle' not in st.session_state:
        st.session_state.puzzle = np.zeros((9, 9), dtype=int)
    if 'original_puzzle' not in st.session_state:
        st.session_state.original_puzzle = np.zeros((9, 9), dtype=int)
    if 'solution' not in st.session_state:
        st.session_state.solution = None
    if 'solving_stats' not in st.session_state:
        st.session_state.solving_stats = None

init_session_state()

# Sidebar
st.sidebar.title("🧮 Sudoku Solver")
st.sidebar.markdown("### Discrete Mathematics Project")
st.sidebar.markdown("---")

# Mode selection
mode = st.sidebar.selectbox(
    "Select Mode",
    ["Solve Puzzle", "Generate Puzzle", "Learn Concepts", "Train Models"]
)

st.sidebar.markdown("---")
st.sidebar.markdown("""
### About
This application demonstrates:
- **Backtracking Algorithm**
- **Predicate Logic**
- **Set Theory**
- **Graph Theory**
- **Machine Learning**
- **Constraint Satisfaction**
""")


# Helper functions
def create_sudoku_grid(board: np.ndarray, editable: bool = True, key_prefix: str = "cell") -> np.ndarray:
    """Creates interactive Sudoku grid"""
    new_board = board.copy()
    
    st.markdown("### Sudoku Grid")
    
    for box_row in range(3):
        cols = st.columns([1, 1, 1])
        for box_col in range(3):
            with cols[box_col]:
                for row in range(box_row * 3, (box_row + 1) * 3):
                    row_cols = st.columns(3)
                    for col in range(box_col * 3, (box_col + 1) * 3):
                        with row_cols[col % 3]:
                            cell_value = int(board[row, col]) if board[row, col] != 0 else 0
                            
                            if editable:
                                value = st.number_input(
                                    f"",
                                    min_value=0,
                                    max_value=9,
                                    value=cell_value,
                                    step=1,
                                    key=f"{key_prefix}_{row}_{col}",
                                    label_visibility="collapsed"
                                )
                                new_board[row, col] = value
                            else:
                                # Display only
                                if cell_value == 0:
                                    st.markdown(f"<div style='text-align: center; padding: 10px;'>·</div>", 
                                              unsafe_allow_html=True)
                                else:
                                    color = "black" if st.session_state.original_puzzle[row, col] != 0 else "green"
                                    st.markdown(f"<div style='text-align: center; padding: 10px; color: {color}; font-weight: bold;'>{cell_value}</div>", 
                                              unsafe_allow_html=True)
                st.markdown("---")
    
    return new_board


def display_board_simple(board: np.ndarray, original: np.ndarray = None):
    """Displays board in simple text format"""
    visualizer = SudokuVisualizer()
    fig = visualizer.create_board_figure(board, original)
    st.plotly_chart(fig, use_container_width=True)


# Main content based on mode
if mode == "Solve Puzzle":
    st.markdown('<div class="main-header">🎯 Solve Sudoku Puzzle</div>', unsafe_allow_html=True)
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.markdown("### Input Puzzle")
        
        # Input method selection
        input_method = st.radio("Input Method", ["Manual Entry", "Load Example", "Random Generate"])
        
        if input_method == "Manual Entry":
            st.info("Enter numbers 1-9 in cells. Use 0 for empty cells.")
            st.session_state.puzzle = create_sudoku_grid(st.session_state.puzzle, editable=True, key_prefix="input")
            
            if st.button("Set as Original"):
                st.session_state.original_puzzle = st.session_state.puzzle.copy()
                st.success("Puzzle saved!")
        
        elif input_method == "Load Example":
            example = st.selectbox("Select Example", ["Easy", "Medium", "Hard"])
            
            examples = {
                "Easy": np.array([
                    [5, 3, 0, 0, 7, 0, 0, 0, 0],
                    [6, 0, 0, 1, 9, 5, 0, 0, 0],
                    [0, 9, 8, 0, 0, 0, 0, 6, 0],
                    [8, 0, 0, 0, 6, 0, 0, 0, 3],
                    [4, 0, 0, 8, 0, 3, 0, 0, 1],
                    [7, 0, 0, 0, 2, 0, 0, 0, 6],
                    [0, 6, 0, 0, 0, 0, 2, 8, 0],
                    [0, 0, 0, 4, 1, 9, 0, 0, 5],
                    [0, 0, 0, 0, 8, 0, 0, 7, 9]
                ]),
                "Medium": np.array([
                    [0, 0, 0, 6, 0, 0, 4, 0, 0],
                    [7, 0, 0, 0, 0, 3, 6, 0, 0],
                    [0, 0, 0, 0, 9, 1, 0, 8, 0],
                    [0, 0, 0, 0, 0, 0, 0, 0, 0],
                    [0, 5, 0, 1, 8, 0, 0, 0, 3],
                    [0, 0, 0, 3, 0, 6, 0, 4, 5],
                    [0, 4, 0, 2, 0, 0, 0, 6, 0],
                    [9, 0, 3, 0, 0, 0, 0, 0, 0],
                    [0, 2, 0, 0, 0, 0, 1, 0, 0]
                ]),
                "Hard": np.array([
                    [0, 0, 0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 3, 0, 8, 5],
                    [0, 0, 1, 0, 2, 0, 0, 0, 0],
                    [0, 0, 0, 5, 0, 7, 0, 0, 0],
                    [0, 0, 4, 0, 0, 0, 1, 0, 0],
                    [0, 9, 0, 0, 0, 0, 0, 0, 0],
                    [5, 0, 0, 0, 0, 0, 0, 7, 3],
                    [0, 0, 2, 0, 1, 0, 0, 0, 0],
                    [0, 0, 0, 0, 4, 0, 0, 0, 9]
                ])
            }
            
            if st.button("Load Example"):
                st.session_state.puzzle = examples[example].copy()
                st.session_state.original_puzzle = examples[example].copy()
                st.success(f"{example} puzzle loaded!")
                st.rerun()
        
        else:  # Random Generate
            difficulty = st.selectbox("Difficulty", ["easy", "medium", "hard", "expert"])
            
            if st.button("Generate Random Puzzle"):
                with st.spinner("Generating puzzle..."):
                    generator = PuzzleGenerator()
                    puzzle, solution = generator.generate_puzzle(difficulty)
                    st.session_state.puzzle = puzzle.copy()
                    st.session_state.original_puzzle = puzzle.copy()
                    st.success("Puzzle generated!")
                    st.rerun()
        
        # Display current puzzle
        st.markdown("### Current Puzzle")
        display_board_simple(st.session_state.puzzle, st.session_state.original_puzzle)
    
    with col2:
        st.markdown("### Solve")
        
        # Solver selection
        solver_type = st.selectbox(
            "Select Solver",
            ["Backtracking (Classical)", "Optimized Backtracking", "Neural Network (ML)"]
        )
        
        # Solve button
        if st.button("🚀 Solve Puzzle", type="primary"):
            if np.sum(st.session_state.original_puzzle) == 0:
                st.error("Please input a puzzle first!")
            else:
                with st.spinner("Solving..."):
                    start_time = time.time()
                    
                    if solver_type == "Backtracking (Classical)":
                        solver = BacktrackingSolver()
                        success, solution = solver.solve(st.session_state.original_puzzle.copy(), use_propagation=False)
                        stats = solver.get_statistics()
                    
                    elif solver_type == "Optimized Backtracking":
                        solver = OptimizedBacktrackingSolver()
                        success, solution = solver.solve(st.session_state.original_puzzle.copy())
                        stats = solver.get_statistics()
                    
                    else:  # Neural Network
                        solver = NeuralSudokuSolver()
                        try:
                            success, solution = solver.solve_iterative(st.session_state.original_puzzle.copy())
                            stats = {'solving_time': time.time() - start_time, 'method': 'Neural Network'}
                        except:
                            st.error("Neural network not trained. Using backtracking fallback.")
                            solver = BacktrackingSolver()
                            success, solution = solver.solve(st.session_state.original_puzzle.copy())
                            stats = solver.get_statistics()
                    
                    if success:
                        st.session_state.solution = solution
                        st.session_state.solving_stats = stats
                        st.success("✅ Puzzle solved successfully!")
                    else:
                        st.error("❌ Could not solve puzzle. Please check input.")
        
        # Display solution
        if st.session_state.solution is not None:
            st.markdown("### Solution")
            display_board_simple(st.session_state.solution, st.session_state.original_puzzle)
            
            # Validate solution
            validator = SolutionValidator()
            is_valid, errors = validator.validate_solution(st.session_state.solution)
            
            if is_valid:
                st.success("✅ Solution is valid!")
            else:
                st.error("❌ Solution has errors:")
                for error in errors:
                    st.write(f"- {error}")
            
            # Display statistics
            if st.session_state.solving_stats:
                st.markdown("### Solving Statistics")
                stats = st.session_state.solving_stats
                
                col_a, col_b, col_c = st.columns(3)
                with col_a:
                    st.metric("Time", f"{stats.get('solving_time', 0):.4f}s")
                with col_b:
                    st.metric("Recursions", stats.get('recursion_count', 'N/A'))
                with col_c:
                    st.metric("Backtracks", stats.get('backtrack_count', 'N/A'))


elif mode == "Generate Puzzle":
    st.markdown('<div class="main-header">🎲 Generate Sudoku Puzzle</div>', unsafe_allow_html=True)
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.markdown("### Generation Options")
        
        difficulty = st.selectbox("Difficulty Level", ["easy", "medium", "hard", "expert"])
        
        pattern_type = st.selectbox(
            "Pattern Type",
            ["Standard", "Symmetric", "Diagonal", "Cross"]
        )
        
        if st.button("🎲 Generate Puzzle", type="primary"):
            with st.spinner("Generating puzzle..."):
                if pattern_type == "Standard":
                    generator = PuzzleGenerator()
                    puzzle, solution = generator.generate_puzzle(difficulty)
                
                elif pattern_type == "Symmetric":
                    generator = PuzzleGenerator()
                    puzzle, solution = generator.generate_symmetric_puzzle(difficulty)
                
                elif pattern_type == "Diagonal":
                    pattern_gen = PatternGenerator()
                    puzzle, solution = pattern_gen.generate_diagonal_pattern()
                
                else:  # Cross
                    pattern_gen = PatternGenerator()
                    puzzle, solution = pattern_gen.generate_cross_pattern()
                
                st.session_state.puzzle = puzzle.copy()
                st.session_state.original_puzzle = puzzle.copy()
                st.session_state.solution = solution.copy()
                
                st.success("Puzzle generated!")
        
        # Analyze difficulty
        if st.button("📊 Analyze Difficulty"):
            if np.sum(st.session_state.puzzle) > 0:
                with st.spinner("Analyzing..."):
                    analyzer = DifficultyAnalyzer()
                    analysis = analyzer.analyze_difficulty(st.session_state.puzzle)
                    
                    st.markdown("### Difficulty Analysis")
                    st.write(f"**Level:** {analysis['difficulty_level']}")
                    st.write(f"**Score:** {analysis['difficulty_score']:.2f}/100")
                    
                    st.markdown("#### Metrics")
                    metrics_df = pd.DataFrame({
                        'Metric': ['Given Cells', 'Empty Cells', 'Avg Options', 'Backtracking Required'],
                        'Value': [
                            analysis['given_cells'],
                            analysis['empty_cells'],
                            f"{analysis['avg_options']:.2f}",
                            analysis['backtracking_required']
                        ]
                    })
                    st.table(metrics_df)
            else:
                st.warning("Generate a puzzle first!")
    
    with col2:
        st.markdown("### Generated Puzzle")
        
        if np.sum(st.session_state.puzzle) > 0:
            display_board_simple(st.session_state.puzzle, st.session_state.original_puzzle)
            
            # Show/hide solution
            show_solution = st.checkbox("Show Solution")
            
            if show_solution and st.session_state.solution is not None:
                st.markdown("### Solution")
                display_board_simple(st.session_state.solution, st.session_state.original_puzzle)
            
            # Export options
            st.markdown("### Export")
            
            if st.button("📋 Copy as Array"):
                array_str = np.array2string(st.session_state.puzzle, separator=', ')
                st.code(array_str, language='python')
                st.info("Copy the array above to use in your code!")
        else:
            st.info("Click 'Generate Puzzle' to create a new puzzle")


elif mode == "Learn Concepts":
    st.markdown('<div class="main-header">📚 Learn Discrete Mathematics Concepts</div>', unsafe_allow_html=True)
    
    concept = st.selectbox(
        "Select Concept",
        ["Backtracking", "Predicate Logic", "Set Theory", "Graph Theory", "Boolean Algebra", "Combinatorics"]
    )
    
    if concept == "Backtracking":
        st.markdown("## Backtracking Algorithm")
        st.markdown("""
        ### What is Backtracking?
        Backtracking is a **recursive algorithm** that tries to build a solution incrementally,
        abandoning solutions that fail to satisfy constraints.
        
        ### How it works in Sudoku:
        1. **Find empty cell** - Locate next cell to fill
        2. **Try numbers 1-9** - Test each number
        3. **Check constraints** - Verify row, column, box rules
        4. **Recurse** - If valid, move to next cell
        5. **Backtrack** - If stuck, undo and try different number
        
        ### Time Complexity:
        - **Worst case:** O(9^(n×n)) where n=9
        - **Average case:** Much better with pruning
        
        ### Discrete Math Concepts:
        - **Recursion:** Function calls itself with smaller problem
        - **State Space Tree:** Tree of all possible configurations
        - **Depth-First Search:** Explores one path completely before backtracking
        - **Pruning:** Eliminates invalid branches early
        """)
        
        # Interactive demo
        st.markdown("### Interactive Demo")
        
        demo_puzzle = np.array([
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
        
        if st.button("Run Backtracking Demo"):
            with st.spinner("Solving with backtracking..."):
                solver = BacktrackingSolver()
                success, solution = solver.solve(demo_puzzle.copy(), use_propagation=False)
                stats = solver.get_statistics()
                
                col1, col2 = st.columns(2)
                with col1:
                    st.markdown("**Original**")
                    display_board_simple(demo_puzzle, demo_puzzle)
                with col2:
                    st.markdown("**Solved**")
                    display_board_simple(solution, demo_puzzle)
                
                st.markdown("### Statistics")
                st.write(f"- Recursion calls: {stats['recursion_count']}")
                st.write(f"- Backtracks: {stats['backtrack_count']}")
                st.write(f"- Time: {stats['solving_time']:.4f} seconds")
    
    elif concept == "Predicate Logic":
        st.markdown("## Predicate Logic in Sudoku")
        st.markdown("""
        ### What is Predicate Logic?
        Predicate logic uses **logical statements** (predicates) that can be true or false.
        
        ### Sudoku Rules as Predicates:
        
        1. **Row Constraint:**
           ```
           ∀i,j,k (i≠k → cell[i][j] ≠ cell[k][j])
           ```
           "For all rows i,k and column j, if i≠k, then cells must be different"
        
        2. **Column Constraint:**
           ```
           ∀i,j,k (j≠k → cell[i][j] ≠ cell[i][k])
           ```
           "For all row i and columns j,k, if j≠k, then cells must be different"
        
        3. **Box Constraint:**
           ```
           ∀cells in box (all different)
           ```
           "All cells in a 3×3 box must have different values"
        
        4. **Domain Constraint:**
           ```
           ∀i,j (1 ≤ cell[i][j] ≤ 9)
           ```
           "All cells must contain values between 1 and 9"
        
        ### Combined Constraint:
        A placement is valid if **ALL** predicates are true:
        ```
        Valid = P_row ∧ P_col ∧ P_box ∧ P_domain
        ```
        (∧ means AND)
        """)
        
        # Interactive demo
        st.markdown("### Test Predicate Logic")
        
        test_board = np.array([
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
        
        col1, col2, col3 = st.columns(3)
        with col1:
            test_row = st.number_input("Row (0-8)", 0, 8, 0)
        with col2:
            test_col = st.number_input("Column (0-8)", 0, 8, 2)
        with col3:
            test_num = st.number_input("Number (1-9)", 1, 9, 4)
        
        if st.button("Check Validity"):
            checker = ConstraintChecker()
            predicate = PredicateLogic()
            
            row_valid = predicate.row_constraint(test_board, test_row, test_num)
            col_valid = predicate.column_constraint(test_board, test_col, test_num)
            box_valid = predicate.box_constraint(test_board, test_row, test_col, test_num)
            all_valid = checker.is_valid_placement(test_board, test_row, test_col, test_num)
            
            st.markdown("### Results")
            st.write(f"✅ Row constraint: {row_valid}" if row_valid else f"❌ Row constraint: {row_valid}")
            st.write(f"✅ Column constraint: {col_valid}" if col_valid else f"❌ Column constraint: {col_valid}")
            st.write(f"✅ Box constraint: {box_valid}" if box_valid else f"❌ Box constraint: {box_valid}")
            st.write(f"**Overall: {'✅ Valid' if all_valid else '❌ Invalid'}**")
    
    elif concept == "Set Theory":
        st.markdown("## Set Theory in Sudoku")
        st.markdown("""
        ### What is Set Theory?
        Set theory deals with **collections of objects** and operations on them.
        
        ### Sets in Sudoku:
        
        1. **Universal Set U:**
           ```
           U = {1, 2, 3, 4, 5, 6, 7, 8, 9}
           ```
           All valid Sudoku numbers
        
        2. **Used Numbers:**
           - Row_Used = numbers already in row
           - Col_Used = numbers already in column
           - Box_Used = numbers already in 3×3 box
        
        3. **Available Numbers:**
           ```
           Available = U - (Row_Used ∪ Col_Used ∪ Box_Used)
           ```
           Set difference: Universal set minus all used numbers
        
        ### Set Operations:
        - **Union (∪):** Combines sets
        - **Intersection (∩):** Common elements
        - **Difference (-):** Elements in first but not second
        - **Cardinality (|A|):** Number of elements in set
        
        ### Example:
        ```
        U = {1,2,3,4,5,6,7,8,9}
        Row_Used = {1,3,5,7,9}
        Col_Used = {2,4,6,8}
        Box_Used = {1,2,3}
        
        All_Used = {1,2,3,4,5,6,7,8,9}  (union)
        Available = {}  (empty - no valid numbers!)
        ```
        """)
        
        # Interactive demo
        st.markdown("### Interactive Set Operations")
        
        set_theory = SetTheory()
        
        st.write("**Universal Set:**", set_theory.universal_set)
        
        used_input = st.text_input("Enter used numbers (comma-separated)", "1,3,5,7")
        
        if st.button("Calculate Available"):
            try:
                used = set(map(int, used_input.split(',')))
                available = set_theory.get_available_numbers(used)
                
                st.write(f"**Used Set:** {used}")
                st.write(f"**Available Set:** {available}")
                st.write(f"**Cardinality:** |Available| = {len(available)}")
            except:
                st.error("Invalid input. Use format: 1,3,5,7")
    
    elif concept == "Graph Theory":
        st.markdown("## Graph Theory in Sudoku")
        st.markdown("""
        ### Sudoku as a Graph
        
        Sudoku can be modeled as a **graph coloring problem**:
        
        - **Vertices (V):** 81 cells in the grid
        - **Edges (E):** Constraints between cells
        - **Colors:** Numbers 1-9
        - **Goal:** Color vertices so no adjacent vertices have same color
        
        ### Graph Properties:
        
        1. **Degree of Vertex:**
           - Each cell has **20 neighbors**
           - 8 in same row + 8 in same column + 4 in same box
        
        2. **Chromatic Number χ(G):**
           - Minimum colors needed = **9**
           - Must use all 9 numbers
        
        3. **Constraint Graph:**
           - Two cells connected if they share row, column, or box
           - Valid solution = valid graph coloring
        
        ### Why This Matters:
        - Helps understand problem structure
        - Enables graph algorithms
        - Shows relationship to other problems
        """)
        
        # Interactive demo
        st.markdown("### Cell Neighbors")
        
        graph = GraphTheory()
        
        col1, col2 = st.columns(2)
        with col1:
            demo_row = st.slider("Row", 0, 8, 4)
        with col2:
            demo_col = st.slider("Column", 0, 8, 4)
        
        neighbors = graph.get_neighbors(demo_row, demo_col)
        degree = graph.degree_of_vertex(demo_row, demo_col)
        
        st.write(f"**Cell ({demo_row}, {demo_col}) has {degree} neighbors:**")
        st.write(f"Neighbors: {neighbors[:10]}... (showing first 10)")
        
        # Visualize on grid
        grid_viz = np.zeros((9, 9))
        grid_viz[demo_row, demo_col] = 2  # Current cell
        for r, c in neighbors:
            grid_viz[r, c] = 1  # Neighbors
        
        st.markdown("**Visualization:** (2=selected cell, 1=neighbors)")
        st.write(grid_viz.astype(int))
    
    elif concept == "Boolean Algebra":
        st.markdown("## Boolean Algebra in Sudoku")
        st.markdown("""
        ### What is Boolean Algebra?
        Boolean algebra deals with **true/false values** and logical operations.
        
        ### Logical Operations:
        
        1. **AND (∧):** True only if both are true
           ```
           True ∧ True = True
           True ∧ False = False
           ```
        
        2. **OR (∨):** True if at least one is true
           ```
           True ∨ False = True
           False ∨ False = False
           ```
        
        3. **NOT (¬):** Opposite value
           ```
           ¬True = False
           ¬False = True
           ```
        
        ### In Sudoku:
        
        Valid placement requires **ALL** constraints to be true:
        ```
        Valid = row_ok AND col_ok AND box_ok AND range_ok
        ```
        
        If **ANY** constraint is false, placement is invalid:
        ```
        Invalid = NOT(row_ok) OR NOT(col_ok) OR NOT(box_ok)
        ```
        """)
        
        # Interactive truth tables
        st.markdown("### Interactive Truth Tables")
        
        bool_alg = BooleanAlgebra()
        
        col1, col2 = st.columns(2)
        with col1:
            p = st.checkbox("P (first condition)", value=True)
        with col2:
            q = st.checkbox("Q (second condition)", value=False)
        
        st.markdown("### Results")
        st.write(f"**P AND Q:** {bool_alg.conjunction(p, q)}")
        st.write(f"**P OR Q:** {bool_alg.disjunction(p, q)}")
        st.write(f"**NOT P:** {bool_alg.negation(p)}")
        st.write(f"**P → Q (implication):** {bool_alg.implication(p, q)}")
        st.write(f"**P ↔ Q (biconditional):** {bool_alg.biconditional(p, q)}")
    
    else:  # Combinatorics
        st.markdown("## Combinatorics in Sudoku")
        st.markdown("""
        ### What is Combinatorics?
        Combinatorics is the study of **counting, arranging, and combining** objects.
        
        ### Key Concepts:
        
        1. **Factorial (n!):**
           ```
           n! = n × (n-1) × (n-2) × ... × 1
           9! = 362,880
           ```
           Ways to arrange n objects
        
        2. **Permutation P(n,r):**
           ```
           P(n,r) = n! / (n-r)!
           ```
           Ways to arrange r objects from n (order matters)
        
        3. **Combination C(n,r):**
           ```
           C(n,r) = n! / (r! × (n-r)!)
           ```
           Ways to choose r objects from n (order doesn't matter)
        
        ### In Sudoku:
        
        - **Total valid grids:** ~6.67 × 10²¹
        - **Solution space:** 9^(empty_cells) upper bound
        - **Puzzle generation:** Combinatorial sampling
        
        ### Example:
        - Ways to fill 3 cells from 9 numbers: P(9,3) = 504
        - Ways to choose 3 numbers from 9: C(9,3) = 84
        """)
        
        # Interactive calculator
        st.markdown("### Combinatorics Calculator")
        
        comb = Combinatorics()
        
        calc_type = st.selectbox("Calculation", ["Factorial", "Permutation", "Combination"])
        
        if calc_type == "Factorial":
            n = st.number_input("n", 0, 12, 9)
            result = comb.factorial(n)
            st.write(f"**{n}! = {result:,}**")
        
        elif calc_type == "Permutation":
            col1, col2 = st.columns(2)
            with col1:
                n = st.number_input("n (total)", 1, 12, 9)
            with col2:
                r = st.number_input("r (select)", 0, n, 3)
            result = comb.permutation(n, r)
            st.write(f"**P({n},{r}) = {result:,}**")
            st.write(f"Ways to arrange {r} items from {n} items")
        
        else:  # Combination
            col1, col2 = st.columns(2)
            with col1:
                n = st.number_input("n (total)", 1, 12, 9)
            with col2:
                r = st.number_input("r (select)", 0, n, 3)
            result = comb.combination(n, r)
            st.write(f"**C({n},{r}) = {result:,}**")
            st.write(f"Ways to choose {r} items from {n} items")


else:  # Train Models
    st.markdown('<div class="main-header">🤖 Train ML Models</div>', unsafe_allow_html=True)
    
    st.warning("⚠️ Model training requires significant computational resources and time.")
    
    model_type = st.selectbox("Select Model", ["Difficulty Classifier", "Neural Solver"])
    
    if model_type == "Difficulty Classifier":
        st.markdown("### Train Difficulty Classifier")
        st.markdown("""
        This trains a Random Forest classifier to predict puzzle difficulty based on features like:
        - Number of given cells
        - Constraint density
        - Option distribution
        - Structural patterns
        """)
        
        n_samples = st.slider("Samples per difficulty", 10, 100, 25)
        
        if st.button("🚀 Start Training"):
            from ml_model.model_trainer import ModelTrainer
            
            with st.spinner("Training classifier... This may take a few minutes."):
                trainer = ModelTrainer()
                
                progress_bar = st.progress(0)
                status_text = st.empty()
                
                status_text.text("Generating training data...")
                progress_bar.progress(20)
                
                metrics = trainer.train_difficulty_classifier(
                    n_per_difficulty=n_samples,
                    model_save_path="difficulty_classifier.pkl"
                )
                
                progress_bar.progress(100)
                status_text.text("Training complete!")
                
                st.success("✅ Model trained successfully!")
                
                st.markdown("### Training Results")
                col1, col2 = st.columns(2)
                with col1:
                    st.metric("Train Accuracy", f"{metrics['train_accuracy']:.2%}")
                with col2:
                    st.metric("Test Accuracy", f"{metrics['test_accuracy']:.2%}")
                
                st.write(f"Training time: {metrics['training_time']:.2f} seconds")
                st.write(f"Total samples: {metrics['n_samples']}")
                
                st.info("Model saved as 'difficulty_classifier.pkl'")
    
    else:  # Neural Solver
        st.markdown("### Train Neural Solver")
        st.markdown("""
        This trains a Convolutional Neural Network to solve Sudoku puzzles.
        
        **Note:** For good performance, the model needs 10,000+ training samples and multiple epochs.
        This demo uses smaller numbers for demonstration purposes.
        """)
        
        n_samples = st.slider("Training samples", 100, 1000, 500, step=100)
        epochs = st.slider("Epochs", 1, 10, 5)
        
        st.warning(f"⚠️ Training {n_samples} samples for {epochs} epochs will take approximately {n_samples * epochs / 100:.0f} minutes.")
        
        if st.button("🚀 Start Training"):
            from ml_model.model_trainer import ModelTrainer
            
            with st.spinner("Training neural network... This will take a while."):
                trainer = ModelTrainer()
                
                progress_bar = st.progress(0)
                status_text = st.empty()
                
                status_text.text("Generating training data...")
                progress_bar.progress(10)
                
                metrics = trainer.train_neural_solver(
                    n_samples=n_samples,
                    epochs=epochs,
                    model_save_path="neural_solver.h5"
                )
                
                progress_bar.progress(100)
                status_text.text("Training complete!")
                
                st.success("✅ Model trained successfully!")
                
                st.markdown("### Training Results")
                col1, col2 = st.columns(2)
                with col1:
                    st.metric("Final Accuracy", f"{metrics['final_accuracy']:.2%}")
                with col2:
                    st.metric("Validation Accuracy", f"{metrics['final_val_accuracy']:.2%}")
                
                st.write(f"Training time: {metrics['training_time']:.2f} seconds")
                st.write(f"Final loss: {metrics['final_loss']:.4f}")
                
                st.info("Model saved as 'neural_solver.h5'")

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: gray;'>
    <p>Sudoku Solver - Discrete Mathematics Project</p>
    <p>Demonstrating: Backtracking • Predicate Logic • Set Theory • Graph Theory • Machine Learning</p>
</div>
""", unsafe_allow_html=True)
