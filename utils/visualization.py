"""
Visualization Utilities for Sudoku Solver
=========================================

This module provides visualization functions for:
1. Sudoku board display
2. Solving process animation
3. Statistics and metrics
4. Constraint visualization

Author: Discrete Mathematics Project
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import List, Tuple
import plotly.graph_objects as go
import plotly.express as px

# Optional imports
try:
    import seaborn as sns
except ImportError:
    sns = None


class SudokuVisualizer:
    """
    Handles all visualization aspects of Sudoku solving
    """
    
    def __init__(self):
        self.colors = {
            'given': '#2C3E50',      # Dark blue for given numbers
            'solved': '#27AE60',     # Green for solved numbers
            'empty': '#ECF0F1',      # Light gray for empty cells
            'highlight': '#E74C3C',  # Red for current cell
            'grid': '#34495E'        # Dark gray for grid lines
        }
    
    def create_board_figure(self, board: np.ndarray, original_board: np.ndarray = None,
                           highlight_cell: Tuple[int, int] = None) -> go.Figure:
        """
        Creates an interactive Plotly figure of Sudoku board
        
        Args:
            board: Current board state
            original_board: Original puzzle (to distinguish given vs solved)
            highlight_cell: Cell to highlight (row, col)
        
        Returns:
            Plotly figure object
        """
        fig = go.Figure()
        
        # Create grid background
        for i in range(10):
            # Thicker lines for 3x3 boxes
            width = 3 if i % 3 == 0 else 1
            color = 'black' if i % 3 == 0 else 'gray'
            
            # Vertical lines
            fig.add_shape(type="line",
                         x0=i, y0=0, x1=i, y1=9,
                         line=dict(color=color, width=width))
            
            # Horizontal lines
            fig.add_shape(type="line",
                         x0=0, y0=i, x1=9, y1=i,
                         line=dict(color=color, width=width))
        
        # Add numbers
        for row in range(9):
            for col in range(9):
                if board[row, col] != 0:
                    # Determine if this is a given number or solved
                    is_given = (original_board is not None and 
                               original_board[row, col] != 0)
                    
                    color = 'black' if is_given else 'green'
                    font_weight = 'bold' if is_given else 'normal'
                    
                    fig.add_annotation(
                        x=col + 0.5,
                        y=8.5 - row,  # Flip y-axis
                        text=str(board[row, col]),
                        showarrow=False,
                        font=dict(size=20, color=color, family="Arial Black" if is_given else "Arial")
                    )
        
        # Highlight cell if specified
        if highlight_cell:
            row, col = highlight_cell
            fig.add_shape(type="rect",
                         x0=col, y0=8-row, x1=col+1, y1=9-row,
                         fillcolor="yellow", opacity=0.3,
                         line=dict(color="red", width=2))
        
        # Configure layout
        fig.update_layout(
            width=500,
            height=500,
            xaxis=dict(range=[0, 9], showticklabels=False, showgrid=False),
            yaxis=dict(range=[0, 9], showticklabels=False, showgrid=False),
            plot_bgcolor='white',
            margin=dict(l=20, r=20, t=20, b=20)
        )
        
        return fig
    
    def create_solving_steps_chart(self, steps: List[int]) -> go.Figure:
        """
        Creates a chart showing solving progress over time
        
        Args:
            steps: List of filled cells count at each step
        
        Returns:
            Plotly figure
        """
        fig = go.Figure()
        
        fig.add_trace(go.Scatter(
            x=list(range(len(steps))),
            y=steps,
            mode='lines+markers',
            name='Filled Cells',
            line=dict(color='#3498DB', width=2),
            marker=dict(size=6)
        ))
        
        fig.update_layout(
            title='Solving Progress',
            xaxis_title='Step Number',
            yaxis_title='Filled Cells',
            hovermode='x unified',
            template='plotly_white'
        )
        
        return fig
    
    def create_difficulty_distribution(self, difficulty_scores: dict) -> go.Figure:
        """
        Creates a bar chart showing difficulty metrics
        
        Args:
            difficulty_scores: Dictionary with difficulty metrics
        
        Returns:
            Plotly figure
        """
        metrics = list(difficulty_scores.keys())
        values = list(difficulty_scores.values())
        
        fig = go.Figure(data=[
            go.Bar(x=metrics, y=values, marker_color='#9B59B6')
        ])
        
        fig.update_layout(
            title='Puzzle Difficulty Metrics',
            xaxis_title='Metric',
            yaxis_title='Score',
            template='plotly_white'
        )
        
        return fig
    
    def create_heatmap(self, board: np.ndarray, title: str = "Sudoku Board") -> go.Figure:
        """
        Creates a heatmap visualization of the board
        
        Args:
            board: Sudoku board
            title: Chart title
        
        Returns:
            Plotly figure
        """
        fig = go.Figure(data=go.Heatmap(
            z=board,
            colorscale='Viridis',
            showscale=True,
            text=board,
            texttemplate='%{text}',
            textfont={"size": 16}
        ))
        
        fig.update_layout(
            title=title,
            width=500,
            height=500,
            xaxis=dict(showticklabels=False),
            yaxis=dict(showticklabels=False)
        )
        
        return fig
    
    def create_constraint_graph_viz(self, cell_constraints: dict) -> go.Figure:
        """
        Visualizes constraint satisfaction for each cell
        
        Args:
            cell_constraints: Dictionary mapping (row,col) to number of constraints
        
        Returns:
            Plotly figure
        """
        # Create matrix of constraint counts
        constraint_matrix = np.zeros((9, 9))
        for (row, col), count in cell_constraints.items():
            constraint_matrix[row, col] = count
        
        fig = go.Figure(data=go.Heatmap(
            z=constraint_matrix,
            colorscale='RdYlGn_r',
            showscale=True,
            colorbar=dict(title="Constraints")
        ))
        
        fig.update_layout(
            title='Constraint Density Map',
            xaxis_title='Column',
            yaxis_title='Row',
            width=500,
            height=500
        )
        
        return fig
    
    def format_board_text(self, board: np.ndarray) -> str:
        """
        Formats board as text for display
        
        Args:
            board: Sudoku board
        
        Returns:
            Formatted string
        """
        lines = []
        for i in range(9):
            if i % 3 == 0 and i != 0:
                lines.append("------+-------+------")
            
            row = []
            for j in range(9):
                if j % 3 == 0 and j != 0:
                    row.append("| ")
                
                cell = board[i, j]
                row.append(str(cell) if cell != 0 else ".")
                row.append(" ")
            
            lines.append("".join(row))
        
        return "\n".join(lines)
    
    def create_comparison_view(self, original: np.ndarray, solved: np.ndarray) -> go.Figure:
        """
        Creates side-by-side comparison of original and solved puzzles
        
        Args:
            original: Original puzzle
            solved: Solved puzzle
        
        Returns:
            Plotly figure with subplots
        """
        from plotly.subplots import make_subplots
        
        fig = make_subplots(
            rows=1, cols=2,
            subplot_titles=('Original Puzzle', 'Solved Puzzle')
        )
        
        # Add original board
        fig.add_trace(
            go.Heatmap(z=original, colorscale='Blues', showscale=False,
                      text=original, texttemplate='%{text}'),
            row=1, col=1
        )
        
        # Add solved board
        fig.add_trace(
            go.Heatmap(z=solved, colorscale='Greens', showscale=False,
                      text=solved, texttemplate='%{text}'),
            row=1, col=2
        )
        
        fig.update_layout(
            height=400,
            width=900,
            showlegend=False
        )
        
        return fig


class MetricsVisualizer:
    """
    Visualizes performance metrics and statistics
    """
    
    @staticmethod
    def create_performance_chart(metrics: dict) -> go.Figure:
        """
        Creates performance metrics visualization
        
        Args:
            metrics: Dictionary with performance data
        
        Returns:
            Plotly figure
        """
        fig = go.Figure()
        
        categories = list(metrics.keys())
        values = list(metrics.values())
        
        fig.add_trace(go.Bar(
            x=categories,
            y=values,
            marker_color=['#3498DB', '#E74C3C', '#2ECC71', '#F39C12'],
            text=values,
            textposition='auto'
        ))
        
        fig.update_layout(
            title='Solver Performance Metrics',
            xaxis_title='Metric',
            yaxis_title='Value',
            template='plotly_white',
            height=400
        )
        
        return fig
    
    @staticmethod
    def create_algorithm_comparison(results: dict) -> go.Figure:
        """
        Compares different solving algorithms
        
        Args:
            results: Dictionary with algorithm names and their metrics
        
        Returns:
            Plotly figure
        """
        algorithms = list(results.keys())
        times = [results[alg]['time'] for alg in algorithms]
        steps = [results[alg]['steps'] for alg in algorithms]
        
        fig = go.Figure()
        
        fig.add_trace(go.Bar(
            name='Time (ms)',
            x=algorithms,
            y=times,
            marker_color='#3498DB'
        ))
        
        fig.add_trace(go.Bar(
            name='Steps',
            x=algorithms,
            y=steps,
            marker_color='#E74C3C'
        ))
        
        fig.update_layout(
            title='Algorithm Comparison',
            xaxis_title='Algorithm',
            yaxis_title='Value',
            barmode='group',
            template='plotly_white'
        )
        
        return fig


def create_styled_board_html(board: np.ndarray, original: np.ndarray = None) -> str:
    """
    Creates HTML representation of Sudoku board with styling
    
    Args:
        board: Current board state
        original: Original puzzle (optional)
    
    Returns:
        HTML string
    """
    html = '<div style="font-family: monospace; font-size: 18px; line-height: 1.5;">'
    
    for i in range(9):
        if i % 3 == 0 and i != 0:
            html += '<div style="border-top: 2px solid black; margin: 5px 0;"></div>'
        
        html += '<div style="display: flex;">'
        for j in range(9):
            if j % 3 == 0 and j != 0:
                html += '<span style="margin: 0 10px; color: black;">|</span>'
            
            cell = board[i, j]
            is_given = original is not None and original[i, j] != 0
            
            color = 'black' if is_given else 'green'
            weight = 'bold' if is_given else 'normal'
            
            value = str(cell) if cell != 0 else '·'
            html += f'<span style="color: {color}; font-weight: {weight}; margin: 0 5px;">{value}</span>'
        
        html += '</div>'
    
    html += '</div>'
    return html
