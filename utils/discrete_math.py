"""
Discrete Mathematics Concepts Implementation
=============================================

This module implements fundamental discrete mathematics concepts used in Sudoku solving:
1. Set Theory - Operations on valid number sets
2. Predicate Logic - Constraint formulation
3. Boolean Algebra - Logical operations
4. Combinatorics - Counting and permutations
5. Graph Theory - Constraint graph representation

Author: Discrete Mathematics Project
"""

import numpy as np
from typing import List, Set, Tuple, Dict
from itertools import combinations, permutations


class SetTheory:
    """
    SET THEORY IMPLEMENTATION
    ========================
    Discrete Math Concept: Set operations for Sudoku constraints
    
    In Sudoku:
    - Universal Set U = {1, 2, 3, 4, 5, 6, 7, 8, 9}
    - Each row, column, box must contain all elements of U
    - Used sets: valid numbers, used numbers, available numbers
    """
    
    def __init__(self):
        # Universal set for Sudoku (all valid numbers)
        self.universal_set = set(range(1, 10))
    
    def get_available_numbers(self, used_numbers: Set[int]) -> Set[int]:
        """
        Set Difference Operation: U - A
        Returns numbers available for placement
        
        Discrete Math: A' = U - A (complement of A)
        """
        return self.universal_set - used_numbers
    
    def union_sets(self, *sets: Set[int]) -> Set[int]:
        """
        Set Union: A ∪ B ∪ C ...
        Combines all used numbers from multiple constraints
        """
        result = set()
        for s in sets:
            result = result.union(s)
        return result
    
    def intersection_sets(self, *sets: Set[int]) -> Set[int]:
        """
        Set Intersection: A ∩ B ∩ C ...
        Finds common elements across sets
        """
        if not sets:
            return set()
        result = sets[0]
        for s in sets[1:]:
            result = result.intersection(s)
        return result
    
    def is_valid_set(self, number_set: Set[int]) -> bool:
        """
        Validates if a set is a subset of universal set
        Discrete Math: A ⊆ U
        """
        return number_set.issubset(self.universal_set)
    
    def cardinality(self, number_set: Set[int]) -> int:
        """
        Returns cardinality (size) of a set
        Discrete Math: |A| = number of elements in A
        """
        return len(number_set)


class PredicateLogic:
    """
    PREDICATE LOGIC IMPLEMENTATION
    ==============================
    Discrete Math Concept: Logical constraints as predicates
    
    Sudoku Rules as Predicates:
    1. Row Constraint: ∀i,j,k (i≠k → cell[i][j] ≠ cell[k][j])
    2. Column Constraint: ∀i,j,k (j≠k → cell[i][j] ≠ cell[i][k])
    3. Box Constraint: ∀cells in box (all different)
    4. Range Constraint: ∀i,j (1 ≤ cell[i][j] ≤ 9)
    """
    
    @staticmethod
    def row_constraint(board: np.ndarray, row: int, num: int) -> bool:
        """
        Row Predicate: P(row, num) = "num not in row"
        
        Logical Form: ∀j ∈ [0,8] (board[row][j] ≠ num)
        Returns: True if constraint satisfied, False otherwise
        """
        return num not in board[row, :]
    
    @staticmethod
    def column_constraint(board: np.ndarray, col: int, num: int) -> bool:
        """
        Column Predicate: Q(col, num) = "num not in column"
        
        Logical Form: ∀i ∈ [0,8] (board[i][col] ≠ num)
        Returns: True if constraint satisfied, False otherwise
        """
        return num not in board[:, col]
    
    @staticmethod
    def box_constraint(board: np.ndarray, row: int, col: int, num: int) -> bool:
        """
        Box Predicate: R(box, num) = "num not in 3x3 box"
        
        Logical Form: ∀(i,j) ∈ box (board[i][j] ≠ num)
        Box is determined by (row//3, col//3)
        """
        box_row, box_col = 3 * (row // 3), 3 * (col // 3)
        box = board[box_row:box_row + 3, box_col:box_col + 3]
        return num not in box
    
    @staticmethod
    def range_constraint(num: int) -> bool:
        """
        Range Predicate: S(num) = "1 ≤ num ≤ 9"
        
        Logical Form: (num ≥ 1) ∧ (num ≤ 9)
        """
        return 1 <= num <= 9
    
    @staticmethod
    def all_constraints(board: np.ndarray, row: int, col: int, num: int) -> bool:
        """
        Combined Constraint: P ∧ Q ∧ R ∧ S
        
        Logical Form: row_constraint AND column_constraint AND box_constraint AND range_constraint
        This is a conjunction (AND) of all predicates
        
        Returns: True if ALL constraints satisfied (valid placement)
        """
        return (PredicateLogic.range_constraint(num) and
                PredicateLogic.row_constraint(board, row, num) and
                PredicateLogic.column_constraint(board, col, num) and
                PredicateLogic.box_constraint(board, row, col, num))


class BooleanAlgebra:
    """
    BOOLEAN ALGEBRA IMPLEMENTATION
    ==============================
    Discrete Math Concept: Boolean operations on constraints
    
    Truth values: True (1), False (0)
    Operations: AND (∧), OR (∨), NOT (¬), XOR (⊕)
    """
    
    @staticmethod
    def conjunction(*conditions: bool) -> bool:
        """
        Logical AND: p ∧ q ∧ r ...
        Returns True only if ALL conditions are True
        
        Truth table for AND:
        p | q | p∧q
        0 | 0 | 0
        0 | 1 | 0
        1 | 0 | 0
        1 | 1 | 1
        """
        return all(conditions)
    
    @staticmethod
    def disjunction(*conditions: bool) -> bool:
        """
        Logical OR: p ∨ q ∨ r ...
        Returns True if ANY condition is True
        
        Truth table for OR:
        p | q | p∨q
        0 | 0 | 0
        0 | 1 | 1
        1 | 0 | 1
        1 | 1 | 1
        """
        return any(conditions)
    
    @staticmethod
    def negation(condition: bool) -> bool:
        """
        Logical NOT: ¬p
        Returns opposite truth value
        
        Truth table for NOT:
        p | ¬p
        0 | 1
        1 | 0
        """
        return not condition
    
    @staticmethod
    def implication(p: bool, q: bool) -> bool:
        """
        Logical Implication: p → q
        "If p then q"
        
        Truth table:
        p | q | p→q
        0 | 0 | 1
        0 | 1 | 1
        1 | 0 | 0
        1 | 1 | 1
        
        Equivalent to: ¬p ∨ q
        """
        return (not p) or q
    
    @staticmethod
    def biconditional(p: bool, q: bool) -> bool:
        """
        Logical Biconditional: p ↔ q
        "p if and only if q"
        
        Truth table:
        p | q | p↔q
        0 | 0 | 1
        0 | 1 | 0
        1 | 0 | 0
        1 | 1 | 1
        """
        return p == q


class Combinatorics:
    """
    COMBINATORICS IMPLEMENTATION
    ============================
    Discrete Math Concept: Counting, permutations, combinations
    
    Used for:
    - Counting valid Sudoku configurations
    - Puzzle generation strategies
    - Solution space analysis
    """
    
    @staticmethod
    def factorial(n: int) -> int:
        """
        Factorial: n! = n × (n-1) × (n-2) × ... × 1
        
        Used in permutation and combination calculations
        Example: 9! = 362,880 (ways to arrange 9 numbers)
        """
        if n <= 1:
            return 1
        result = 1
        for i in range(2, n + 1):
            result *= i
        return result
    
    @staticmethod
    def permutation(n: int, r: int) -> int:
        """
        Permutation: P(n,r) = n! / (n-r)!
        
        Number of ways to arrange r items from n items (order matters)
        Example: P(9,3) = ways to fill 3 cells from 9 numbers
        """
        if r > n:
            return 0
        return Combinatorics.factorial(n) // Combinatorics.factorial(n - r)
    
    @staticmethod
    def combination(n: int, r: int) -> int:
        """
        Combination: C(n,r) = n! / (r! × (n-r)!)
        
        Number of ways to choose r items from n items (order doesn't matter)
        Example: C(9,3) = ways to select 3 numbers from 9
        """
        if r > n:
            return 0
        return (Combinatorics.factorial(n) // 
                (Combinatorics.factorial(r) * Combinatorics.factorial(n - r)))
    
    @staticmethod
    def count_empty_cells(board: np.ndarray) -> int:
        """
        Counts empty cells (0s) in Sudoku board
        Used to estimate solution space size
        """
        return np.sum(board == 0)
    
    @staticmethod
    def estimate_solution_space(board: np.ndarray) -> float:
        """
        Estimates size of solution space
        
        Rough estimate: 9^(empty_cells)
        Actual space is smaller due to constraints
        """
        empty = Combinatorics.count_empty_cells(board)
        return 9 ** empty


class GraphTheory:
    """
    GRAPH THEORY IMPLEMENTATION
    ===========================
    Discrete Math Concept: Sudoku as constraint graph
    
    Graph Representation:
    - Vertices (V): 81 cells in Sudoku grid
    - Edges (E): Constraints between cells
    - Graph Coloring: Assigning numbers (colors) to vertices
    - Chromatic Number: χ(G) = 9 (minimum colors needed)
    """
    
    def __init__(self):
        self.vertices = 81  # 9x9 grid
        self.colors = 9     # Numbers 1-9
    
    def get_neighbors(self, row: int, col: int) -> List[Tuple[int, int]]:
        """
        Returns all neighbors (constrained cells) for a given cell
        
        Neighbors are cells that:
        1. Share the same row
        2. Share the same column
        3. Share the same 3x3 box
        
        This forms the constraint graph edges
        """
        neighbors = []
        
        # Row neighbors
        for c in range(9):
            if c != col:
                neighbors.append((row, c))
        
        # Column neighbors
        for r in range(9):
            if r != row:
                neighbors.append((r, col))
        
        # Box neighbors
        box_row, box_col = 3 * (row // 3), 3 * (col // 3)
        for r in range(box_row, box_row + 3):
            for c in range(box_col, box_col + 3):
                if (r, c) != (row, col) and (r, c) not in neighbors:
                    neighbors.append((r, c))
        
        return neighbors
    
    def degree_of_vertex(self, row: int, col: int) -> int:
        """
        Degree of a vertex: number of edges connected to it
        
        In Sudoku: Each cell has 20 neighbors
        - 8 in same row
        - 8 in same column
        - 4 in same box (excluding row/column overlaps)
        """
        return len(self.get_neighbors(row, col))
    
    def is_valid_coloring(self, board: np.ndarray) -> bool:
        """
        Checks if current board state is a valid graph coloring
        
        Valid coloring: No two adjacent vertices have same color
        In Sudoku: No two constrained cells have same number
        """
        for row in range(9):
            for col in range(9):
                if board[row, col] != 0:
                    num = board[row, col]
                    # Temporarily remove number to check neighbors
                    board[row, col] = 0
                    if not PredicateLogic.all_constraints(board, row, col, num):
                        board[row, col] = num
                        return False
                    board[row, col] = num
        return True
    
    def chromatic_number(self) -> int:
        """
        Returns chromatic number of Sudoku graph
        
        Chromatic Number χ(G): Minimum colors needed for valid coloring
        For Sudoku: χ(G) = 9 (always need all 9 numbers)
        """
        return 9


# Utility function to demonstrate all concepts
def demonstrate_discrete_math():
    """
    Demonstrates all discrete mathematics concepts with examples
    """
    print("=" * 60)
    print("DISCRETE MATHEMATICS CONCEPTS IN SUDOKU")
    print("=" * 60)
    
    # 1. Set Theory
    print("\n1. SET THEORY")
    print("-" * 60)
    st = SetTheory()
    used = {1, 3, 5, 7, 9}
    available = st.get_available_numbers(used)
    print(f"Universal Set: {st.universal_set}")
    print(f"Used Numbers: {used}")
    print(f"Available Numbers: {available}")
    print(f"Cardinality of Available: {st.cardinality(available)}")
    
    # 2. Predicate Logic
    print("\n2. PREDICATE LOGIC")
    print("-" * 60)
    board = np.zeros((9, 9), dtype=int)
    board[0, :] = [5, 3, 0, 0, 7, 0, 0, 0, 0]
    print(f"Can place 5 at (0,2)? {PredicateLogic.all_constraints(board, 0, 2, 5)}")
    print(f"Can place 4 at (0,2)? {PredicateLogic.all_constraints(board, 0, 2, 4)}")
    
    # 3. Boolean Algebra
    print("\n3. BOOLEAN ALGEBRA")
    print("-" * 60)
    ba = BooleanAlgebra()
    print(f"True AND True = {ba.conjunction(True, True)}")
    print(f"True OR False = {ba.disjunction(True, False)}")
    print(f"NOT True = {ba.negation(True)}")
    
    # 4. Combinatorics
    print("\n4. COMBINATORICS")
    print("-" * 60)
    comb = Combinatorics()
    print(f"9! = {comb.factorial(9)}")
    print(f"P(9,3) = {comb.permutation(9, 3)}")
    print(f"C(9,3) = {comb.combination(9, 3)}")
    
    # 5. Graph Theory
    print("\n5. GRAPH THEORY")
    print("-" * 60)
    gt = GraphTheory()
    neighbors = gt.get_neighbors(0, 0)
    print(f"Neighbors of cell (0,0): {len(neighbors)} cells")
    print(f"Chromatic Number: {gt.chromatic_number()}")
    print("=" * 60)


if __name__ == "__main__":
    demonstrate_discrete_math()
