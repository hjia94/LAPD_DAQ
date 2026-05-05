"""
Motion control and position management package for LAPD DAQ system.

This package contains:
- PositionManager: Handles position arrays, HDF5 position storage, and motor initialization
- Position generation utilities
- Boundary checking functions
"""

# Import main classes and functions for easy access
from .position_manager import (
    PositionManager,
    # Configuration functions
    load_position_config,
    # Position generation functions
    get_positions_xy,
    get_positions_xyz,
    get_positions_45deg,
    create_all_positions_45deg,
    # Boundary checking functions
    outer_boundary,
    obstacle_boundary,
    motor_boundary,
    motor_boundary_2D
)

from .Motor_Control import Motor_Control_2D, Motor_Control_3D
from .Motor_Control_1D import Motor_Control

__all__ = [
    'PositionManager',
    # Configuration functions
    'load_position_config',
    # Position generation functions
    'get_positions_xy',
    'get_positions_xyz',
    'get_positions_45deg',
    'create_all_positions_45deg',
    # Boundary checking functions
    'outer_boundary',
    'obstacle_boundary',
    'motor_boundary',
    'motor_boundary_2D',
    # Motor control classes
    'Motor_Control_2D',
    'Motor_Control_3D',
    'Motor_Control'
] 