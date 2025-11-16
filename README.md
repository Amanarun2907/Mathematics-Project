# Sudoku Solver - Discrete Mathematics Project

## 🎯 Project Overview
A comprehensive Sudoku puzzle solver that demonstrates discrete mathematics concepts including:
- **Backtracking Algorithms** (Recursive problem solving)
- **Predicate Logic** (Constraint satisfaction)
- **Set Theory** (Valid number sets and constraints)
- **Graph Theory** (Sudoku as constraint satisfaction problem)
- **Boolean Algebra** (Logical constraint evaluation)
- **Combinatorics** (Solution counting and puzzle generation)

## 🚀 Features
- **Classical Backtracking Solver** with predicate logic constraints
- **Neural Network Solver** using deep learning
- **Puzzle Generator** with multiple difficulty levels
- **Difficulty Classifier** using pattern recognition
- **Interactive Streamlit UI** for user interaction
- **Solution Validation** and puzzle verification

## 📋 Prerequisites
- Python 3.8 or higher
- pip package manager

## 🔧 Installation

1. Clone or download this project

2. Install dependencies:
```bash
pip install -r requirements.txt
```

## 🎮 Usage

Run the Streamlit application:
```bash
streamlit run app.py
```

The application will open in your default web browser at `http://localhost:8501`

## 📁 Project Structure
```
sudoku-solver/
├── app.py                      # Main Streamlit application
├── solver/
│   ├── backtracking_solver.py  # Classical backtracking algorithm
│   ├── constraint_logic.py     # Predicate logic constraints
│   └── solution_validator.py   # Solution verification
├── generator/
│   ├── puzzle_generator.py     # Sudoku puzzle creation
│   └── difficulty_analyzer.py  # Puzzle difficulty calculation
├── ml_model/
│   ├── neural_solver.py        # Neural network solver
│   ├── difficulty_classifier.py # ML-based difficulty classification
│   └── model_trainer.py        # Training utilities
├── utils/
│   ├── discrete_math.py        # Discrete math concepts implementation
│   └── visualization.py        # Visualization helpers
├── requirements.txt            # Project dependencies
└── README.md                   # This file
```

## 🧮 Discrete Mathematics Concepts Used

### 1. Backtracking (Recursive Algorithms)
- Systematic exploration of solution space
- Depth-first search with pruning
- Time complexity: O(9^(n*n)) worst case

### 2. Predicate Logic
- Constraints as logical predicates
- Row constraint: ∀i,j,k (i≠k → cell[i][j] ≠ cell[k][j])
- Column constraint: ∀i,j,k (j≠k → cell[i][j] ≠ cell[i][k])
- Box constraint: Similar logical formulation

### 3. Set Theory
- Valid number set: {1, 2, 3, 4, 5, 6, 7, 8, 9}
- Constraint sets for rows, columns, and boxes
- Set operations: union, intersection, difference

### 4. Graph Theory
- Sudoku as graph coloring problem
- Vertices: cells, Edges: constraints
- Chromatic number: 9

### 5. Boolean Algebra
- Constraint satisfaction as boolean expressions
- AND/OR operations on constraints
- Truth value evaluation

### 6. Combinatorics
- Counting valid Sudoku grids: ~6.67 × 10^21
- Puzzle generation with unique solutions
- Permutations and combinations

## 🎓 Learning Outcomes
- Understanding constraint satisfaction problems
- Implementing backtracking algorithms
- Applying predicate logic to real problems
- Using neural networks for puzzle solving
- Pattern recognition and classification
- Algorithm optimization techniques

## 📊 Performance
- **Backtracking Solver**: Solves most puzzles in < 1 second
- **Neural Network**: Trained on 1M+ puzzles
- **Generator**: Creates valid puzzles in milliseconds

## 🤝 Contributing
This is an educational project demonstrating discrete mathematics concepts.

## 📝 License
Educational use only.

## 👨‍💻 Author
Discrete Mathematics Project - Sudoku Solver with AI/ML Integration
