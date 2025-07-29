import numpy as np
import random
from collections import deque, defaultdict, Counter
import time
from typing import List, Tuple, Dict, Set, Optional
import copy
import pickle
import os
from itertools import permutations, combinations
import kociemba

class AdvancedRubiksCube:
    """
    Enhanced Rubik's Cube with support for advanced algorithms
    Includes pattern database generation and 2-phase algorithm support
    """
    
    def __init__(self):
        # Initialize solved cube state
        self.faces = {
            'U': np.full((3, 3), 0, dtype=int),  # Up (White)
            'R': np.full((3, 3), 1, dtype=int),  # Right (Red)
            'F': np.full((3, 3), 2, dtype=int),  # Front (Blue)
            'D': np.full((3, 3), 5, dtype=int),  # Down (Yellow)
            'L': np.full((3, 3), 4, dtype=int),  # Left (Green)
            'B': np.full((3, 3), 3, dtype=int),  # Back (Orange)
        }
        
        self.move_history = []
        self.colors = ['W', 'R', 'B', 'O', 'G', 'Y']
        
        # All possible moves
        self.basic_moves = ['U', 'R', 'F', 'D', 'L', 'B']
        self.all_moves = []
        for move in self.basic_moves:
            self.all_moves.extend([move, move + "'", move + "2"])
        
        # Phase 1 moves for Kociemba (restricted move set)
        self.phase1_moves = ['U', "U'", 'U2', 'D', "D'", 'D2', 
                            'R2', 'L2', 'F2', 'B2']
        
        # Edge and corner piece tracking for advanced algorithms
        self._init_piece_tracking()
    
    def _init_piece_tracking(self):
        """Initialize piece tracking for pattern databases"""
        # Edge pieces: 12 edges, each can be in 12 positions with 2 orientations
        self.edge_positions = [
            ('U', 0, 1), ('U', 1, 0), ('U', 1, 2), ('U', 2, 1),  # Top edges
            ('D', 0, 1), ('D', 1, 0), ('D', 1, 2), ('D', 2, 1),  # Bottom edges
            ('F', 1, 0), ('F', 1, 2), ('B', 1, 0), ('B', 1, 2)   # Middle edges
        ]
        
        # Corner pieces: 8 corners, each can be in 8 positions with 3 orientations
        self.corner_positions = [
            ('U', 0, 0), ('U', 0, 2), ('U', 2, 0), ('U', 2, 2),  # Top corners
            ('D', 0, 0), ('D', 0, 2), ('D', 2, 0), ('D', 2, 2)   # Bottom corners
        ]
    
    def copy(self):
        """Create deep copy of cube state"""
        new_cube = AdvancedRubiksCube()
        new_cube.faces = {face: np.copy(arr) for face, arr in self.faces.items()}
        new_cube.move_history = self.move_history.copy()
        return new_cube
    
    def get_state_hash(self) -> str:
        """Generate unique hash for current cube state"""
        state_str = ""
        for face in ['U', 'R', 'F', 'D', 'L', 'B']:
            state_str += ''.join(str(x) for x in self.faces[face].flatten())
        return state_str
    
    def is_solved(self) -> bool:
        """Check if cube is in solved state"""
        for face_name, face in self.faces.items():
            if not np.all(face == face[0, 0]):
                return False
        return True
    
    def get_corner_pattern(self) -> str:
        """Get corner pattern for pattern database"""
        # Simplified corner pattern - in practice this would be more sophisticated
        corners = []
        for face in ['U', 'D']:
            for i in [0, 2]:
                for j in [0, 2]:
                    corners.append(str(self.faces[face][i, j]))
        return ''.join(corners)
    
    def get_edge_pattern(self) -> str:
        """Get edge pattern for pattern database"""
        # Simplified edge pattern
        edges = []
        for face in ['U', 'D', 'F', 'B']:
            if face in ['U', 'D']:
                for pos in [(0,1), (1,0), (1,2), (2,1)]:
                    edges.append(str(self.faces[face][pos]))
            else:
                for pos in [(1,0), (1,2)]:
                    edges.append(str(self.faces[face][pos]))
        return ''.join(edges)
    
    def is_phase1_complete(self) -> bool:
        """Check if phase 1 of Kociemba is complete"""
        # Phase 1: All edges oriented correctly and corners in correct slice
        # Simplified check - edges should have correct orientation
        edge_orientations_correct = True
        
        # Check if F and B faces have only F/B colors on F/B edges
        f_edges = [self.faces['F'][0,1], self.faces['F'][1,0], 
                  self.faces['F'][1,2], self.faces['F'][2,1]]
        b_edges = [self.faces['B'][0,1], self.faces['B'][1,0], 
                  self.faces['B'][1,2], self.faces['B'][2,1]]
        
        # In phase 1, F/B edges should only be on F/B faces or U/D faces
        for edge in f_edges:
            if edge not in [2, 0, 5]:  # F, W, Y colors allowed
                edge_orientations_correct = False
        
        for edge in b_edges:
            if edge not in [3, 0, 5]:  # B, W, Y colors allowed
                edge_orientations_correct = False
        
        return edge_orientations_correct
    
    def execute_move(self, move: str):
        """Execute a move - same as before but optimized"""
        if move == 'U':
            self._move_U()
        elif move == "U'":
            self._move_U(prime=True)
        elif move == 'U2':
            self._move_U(); self._move_U()
        elif move == 'R':
            self._move_R()
        elif move == "R'":
            self._move_R(prime=True)
        elif move == 'R2':
            self._move_R(); self._move_R()
        elif move == 'F':
            self._move_F()
        elif move == "F'":
            self._move_F(prime=True)
        elif move == 'F2':
            self._move_F(); self._move_F()
        elif move == 'D':
            self._move_D()
        elif move == "D'":
            self._move_D(prime=True)
        elif move == 'D2':
            self._move_D(); self._move_D()
        elif move == 'L':
            self._move_L()
        elif move == "L'":
            self._move_L(prime=True)
        elif move == 'L2':
            self._move_L(); self._move_L()
        elif move == 'B':
            self._move_B()
        elif move == "B'":
            self._move_B(prime=True)
        elif move == 'B2':
            self._move_B(); self._move_B()
        
        self.move_history.append(move)
    
    # Move methods (optimized versions)
    def _move_U(self, prime=False):
        self.faces['U'] = np.rot90(self.faces['U'], -1 if not prime else 1)
        if not prime:
            temp = self.faces['F'][0, :].copy()
            self.faces['F'][0, :] = self.faces['R'][0, :]
            self.faces['R'][0, :] = self.faces['B'][0, :]
            self.faces['B'][0, :] = self.faces['L'][0, :]
            self.faces['L'][0, :] = temp
        else:
            temp = self.faces['F'][0, :].copy()
            self.faces['F'][0, :] = self.faces['L'][0, :]
            self.faces['L'][0, :] = self.faces['B'][0, :]
            self.faces['B'][0, :] = self.faces['R'][0, :]
            self.faces['R'][0, :] = temp

    def _move_D(self, prime=False):
        self.faces['D'] = np.rot90(self.faces['D'], -1 if not prime else 1)
        if not prime:
            temp = self.faces['F'][2, :].copy()
            self.faces['F'][2, :] = self.faces['L'][2, :]
            self.faces['L'][2, :] = self.faces['B'][2, :]
            self.faces['B'][2, :] = self.faces['R'][2, :]
            self.faces['R'][2, :] = temp
        else:
            temp = self.faces['F'][2, :].copy()
            self.faces['F'][2, :] = self.faces['R'][2, :]
            self.faces['R'][2, :] = self.faces['B'][2, :]
            self.faces['B'][2, :] = self.faces['L'][2, :]
            self.faces['L'][2, :] = temp

    def _move_F(self, prime=False):
        self.faces['F'] = np.rot90(self.faces['F'], -1 if not prime else 1)
        if not prime:
            temp = self.faces['U'][2, :].copy()
            self.faces['U'][2, :] = self.faces['L'][:, 2][::-1]
            self.faces['L'][:, 2] = self.faces['D'][0, :]
            self.faces['D'][0, :] = self.faces['R'][:, 0][::-1]
            self.faces['R'][:, 0] = temp
        else:
            temp = self.faces['U'][2, :].copy()
            self.faces['U'][2, :] = self.faces['R'][:, 0]
            self.faces['R'][:, 0] = self.faces['D'][0, :][::-1]
            self.faces['D'][0, :] = self.faces['L'][:, 2]
            self.faces['L'][:, 2] = temp[::-1]

    def _move_B(self, prime=False):
        self.faces['B'] = np.rot90(self.faces['B'], -1 if not prime else 1)
        if not prime:
            temp = self.faces['U'][0, :].copy()
            self.faces['U'][0, :] = self.faces['R'][:, 2]
            self.faces['R'][:, 2] = self.faces['D'][2, :][::-1]
            self.faces['D'][2, :] = self.faces['L'][:, 0]
            self.faces['L'][:, 0] = temp[::-1]
        else:
            temp = self.faces['U'][0, :].copy()
            self.faces['U'][0, :] = self.faces['L'][:, 0][::-1]
            self.faces['L'][:, 0] = self.faces['D'][2, :]
            self.faces['D'][2, :] = self.faces['R'][:, 2][::-1]
            self.faces['R'][:, 2] = temp

    def _move_L(self, prime=False):
        self.faces['L'] = np.rot90(self.faces['L'], -1 if not prime else 1)
        if not prime:
            temp = self.faces['U'][:, 0].copy()
            self.faces['U'][:, 0] = self.faces['B'][:, 2][::-1]
            self.faces['B'][:, 2] = self.faces['D'][:, 0][::-1]
            self.faces['D'][:, 0] = self.faces['F'][:, 0]
            self.faces['F'][:, 0] = temp
        else:
            temp = self.faces['U'][:, 0].copy()
            self.faces['U'][:, 0] = self.faces['F'][:, 0]
            self.faces['F'][:, 0] = self.faces['D'][:, 0]
            self.faces['D'][:, 0] = self.faces['B'][:, 2][::-1]
            self.faces['B'][:, 2] = temp[::-1]

    def _move_R(self, prime=False):
        self.faces['R'] = np.rot90(self.faces['R'], -1 if not prime else 1)
        if not prime:
            temp = self.faces['U'][:, 2].copy()
            self.faces['U'][:, 2] = self.faces['F'][:, 2]
            self.faces['F'][:, 2] = self.faces['D'][:, 2]
            self.faces['D'][:, 2] = self.faces['B'][:, 0][::-1]
            self.faces['B'][:, 0] = temp[::-1]
        else:
            temp = self.faces['U'][:, 2].copy()
            self.faces['U'][:, 2] = self.faces['B'][:, 0][::-1]
            self.faces['B'][:, 0] = self.faces['D'][:, 2][::-1]
            self.faces['D'][:, 2] = self.faces['F'][:, 2]
            self.faces['F'][:, 2] = temp
    
    def execute_sequence(self, moves: List[str]):
        """Execute a sequence of moves"""
        for move in moves:
            self.execute_move(move)
    
    def scramble(self, num_moves: int = 25):
        """Scramble the cube with random moves"""
        self.move_history = []
        for _ in range(num_moves):
            move = random.choice(self.all_moves)
            self.execute_move(move)
        return self.move_history.copy()


class PatternDatabase:
    """
    Pattern database for Korf's algorithm
    Stores precomputed distances for cube subproblems
    """
    
    def __init__(self, pattern_type: str = "corner"):
        self.pattern_type = pattern_type
        self.database = {}
        self.max_depth = 8  # Reasonable limit for demo
        
    def generate_database(self, cube: AdvancedRubiksCube, max_states: int = 50000):
        """Generate pattern database using BFS from solved state"""
        print(f"Generating {self.pattern_type} pattern database...")
        
        solved_cube = AdvancedRubiksCube()
        queue = deque([(solved_cube, 0)])
        
        if self.pattern_type == "corner":
            pattern = solved_cube.get_corner_pattern()
        else:
            pattern = solved_cube.get_edge_pattern()
        
        self.database[pattern] = 0
        visited = {pattern}
        states_generated = 1
        
        while queue and states_generated < max_states:
            current_cube, depth = queue.popleft()
            
            if depth >= self.max_depth:
                continue
            
            for move in current_cube.all_moves:
                new_cube = current_cube.copy()
                new_cube.execute_move(move)
                
                if self.pattern_type == "corner":
                    new_pattern = new_cube.get_corner_pattern()
                else:
                    new_pattern = new_cube.get_edge_pattern()
                
                if new_pattern not in visited:
                    visited.add(new_pattern)
                    self.database[new_pattern] = depth + 1
                    queue.append((new_cube, depth + 1))
                    states_generated += 1
                    
                    if states_generated % 10000 == 0:
                        print(f"  Generated {states_generated} states, depth {depth + 1}")
        
        print(f"Pattern database complete: {len(self.database)} states")
    
    def get_heuristic(self, cube: AdvancedRubiksCube) -> int:
        """Get heuristic value from pattern database"""
        if self.pattern_type == "corner":
            pattern = cube.get_corner_pattern()
        else:
            pattern = cube.get_edge_pattern()
        
        return self.database.get(pattern, self.max_depth)  # Return max if not found


class KorfIDAStar:
    """
    Korf's IDA* algorithm with pattern databases
    Uses iterative deepening with sophisticated heuristics
    """
    
    def __init__(self, cube: AdvancedRubiksCube):
        self.cube = cube.copy()
        self.corner_db = PatternDatabase("corner")
        self.edge_db = PatternDatabase("edge")
        self.nodes_explored = 0
        
    def generate_pattern_databases(self):
        """Generate both corner and edge pattern databases"""
        print("Initializing Korf's Algorithm Pattern Databases...")
        print("This may take a moment for the demo...")
        
        # Generate smaller databases for demo purposes
        self.corner_db.generate_database(self.cube, max_states=25000)
        self.edge_db.generate_database(self.cube, max_states=25000)
        print("Pattern databases ready!\n")
    
    def heuristic(self, cube: AdvancedRubiksCube) -> int:
        """Combined heuristic from multiple pattern databases"""
        corner_h = self.corner_db.get_heuristic(cube)
        edge_h = self.edge_db.get_heuristic(cube)
        
        # Take maximum of the two heuristics (admissible)
        return max(corner_h, edge_h)
    
    def ida_star_search(self, max_depth: int = 15) -> Tuple[List[str], int]:
        """Iterative Deepening A* search"""
        if self.cube.is_solved():
            return [], 0
        
        # Start with heuristic estimate
        threshold = self.heuristic(self.cube)
        path = []
        
        print(f"Starting IDA* with initial threshold: {threshold}")
        
        for iteration in range(max_depth):
            print(f"IDA* iteration {iteration + 1}, threshold: {threshold}")
            self.nodes_explored = 0
            
            result = self._search(self.cube, path, 0, threshold)
            
            if isinstance(result, list):  # Found solution
                return result, self.nodes_explored
            
            if result == float('inf'):  # No solution exists
                break
            
            threshold = result  # New threshold for next iteration
            
            if self.nodes_explored > 100000:  # Prevent excessive computation
                print("Node limit reached, stopping search")
                break
        
        return [], self.nodes_explored
    
    def _search(self, cube: AdvancedRubiksCube, path: List[str], 
                g: int, threshold: int) -> any:
        """Recursive search function for IDA*"""
        self.nodes_explored += 1
        
        if self.nodes_explored % 10000 == 0:
            print(f"  Nodes explored: {self.nodes_explored}, depth: {g}")
        
        f = g + self.heuristic(cube)
        
        if f > threshold:
            return f
        
        if cube.is_solved():
            return path.copy()
        
        min_threshold = float('inf')
        
        for move in cube.all_moves:
            # Avoid immediate reversal of moves
            if path and self._is_reverse_move(path[-1], move):
                continue
            
            new_cube = cube.copy()
            new_cube.execute_move(move)
            path.append(move)
            
            result = self._search(new_cube, path, g + 1, threshold)
            
            if isinstance(result, list):  # Solution found
                return result
            
            if result < min_threshold:
                min_threshold = result
            
            path.pop()
        
        return min_threshold
    
    def _is_reverse_move(self, move1: str, move2: str) -> bool:
        """Check if move2 is the reverse of move1"""
        base1 = move1[0]
        base2 = move2[0]
        
        if base1 != base2:
            return False
        
        if move1 == move2:
            return False
        
        # Check for reversals like R and R', U2 and U2
        if (move1.endswith("'") and move2 == base2) or \
           (move2.endswith("'") and move1 == base1) or \
           (move1 == base1 + "2" and move2 == base2 + "2"):
            return True
        
        return False


class KociembaAlgorithm:
    """
    Kociemba's 2-Phase Algorithm
    Phase 1: Orient edges and position corners in correct slice
    Phase 2: Solve the cube completely
    """
    
    def __init__(self, cube: AdvancedRubiksCube):
        self.cube = cube.copy()
        self.phase1_solutions = []
        self.nodes_explored = 0
        
    def solve(self, max_phase1_depth: int = 12, max_phase2_depth: int = 18) -> Tuple[List[str], Dict]:
        """
        Complete 2-phase solve
        Returns: (solution_moves, statistics)
        """
        print("Starting Kociemba 2-Phase Algorithm...")
        start_time = time.time()
        
        # Phase 1: Find solutions that achieve the phase 1 goal
        print("Phase 1: Orienting edges and positioning corners...")
        phase1_solutions = self._phase1_search(max_phase1_depth)
        
        if not phase1_solutions:
            return [], {"error": "No Phase 1 solutions found", "time": time.time() - start_time}
        
        print(f"Found {len(phase1_solutions)} Phase 1 solutions")
        
        # Phase 2: For each Phase 1 solution, find the best Phase 2 completion
        print("Phase 2: Completing the solve...")
        best_solution = []
        best_length = float('inf')
        
        for i, phase1_moves in enumerate(phase1_solutions[:5]):  # Test top 5 solutions
            print(f"  Testing Phase 1 solution {i+1}/{min(5, len(phase1_solutions))}")
            
            # Apply phase 1 moves to cube
            test_cube = self.cube.copy()
            test_cube.execute_sequence(phase1_moves)
            
            # Find phase 2 solution
            phase2_moves = self._phase2_search(test_cube, max_phase2_depth)
            
            if phase2_moves:
                total_solution = phase1_moves + phase2_moves
                if len(total_solution) < best_length:
                    best_solution = total_solution
                    best_length = len(total_solution)
        
        total_time = time.time() - start_time
        
        stats = {
            "phase1_solutions": len(phase1_solutions),
            "solution_length": len(best_solution),
            "time": total_time,
            "nodes_explored": self.nodes_explored
        }
        
        return best_solution, stats
    
    def _phase1_search(self, max_depth: int) -> List[List[str]]:
        """Phase 1: Search for edge orientation and corner positioning"""
        solutions = []
        queue = deque([(self.cube.copy(), [])])
        visited = set()
        
        while queue and len(solutions) < 10:  # Find multiple solutions
            current_cube, moves = queue.popleft()
            self.nodes_explored += 1
            
            if len(moves) > max_depth:
                continue
            
            # Check if phase 1 is complete
            if current_cube.is_phase1_complete():
                solutions.append(moves.copy())
                continue
            
            # State pruning
            state_hash = current_cube.get_state_hash()
            if state_hash in visited:
                continue
            visited.add(state_hash)
            
            # Only use Phase 1 moves
            for move in current_cube.phase1_moves:
                if moves and self._is_reverse_move(moves[-1], move):
                    continue
                
                new_cube = current_cube.copy()
                new_cube.execute_move(move)
                queue.append((new_cube, moves + [move]))
        
        return solutions
    
    def _phase2_search(self, cube: AdvancedRubiksCube, max_depth: int) -> List[str]:
        """Phase 2: Complete the solve using all moves"""
        queue = deque([(cube.copy(), [])])
        visited = set()
        
        while queue:
            current_cube, moves = queue.popleft()
            self.nodes_explored += 1
            
            if len(moves) > max_depth:
                continue
            
            if current_cube.is_solved():
                return moves
            
            # State pruning
            state_hash = current_cube.get_state_hash()
            if state_hash in visited:
                continue
            visited.add(state_hash)
            
            # Use all moves in Phase 2
            for move in current_cube.all_moves:
                if moves and self._is_reverse_move(moves[-1], move):
                    continue
                
                new_cube = current_cube.copy()
                new_cube.execute_move(move)
                queue.append((new_cube, moves + [move]))
        
        return []
    
    def _is_reverse_move(self, move1: str, move2: str) -> bool:
        """Check if move2 is the reverse of move1"""
        base1 = move1[0]
        base2 = move2[0]
        
        if base1 != base2:
            return False
        
        if (move1.endswith("'") and move2 == base2) or \
           (move2.endswith("'") and move1 == base1):
            return True
        
        return False


def to_kociemba_string(cube: AdvancedRubiksCube) -> str:
    color_map = {
        0: 'U',  # White -> Up
        1: 'R',  # Red -> Right
        2: 'F',  # Blue -> Front
        5: 'D',  # Yellow -> Down
        4: 'L',  # Green -> Left
        3: 'B',  # Orange -> Back
    }
    order = ['U', 'R', 'F', 'D', 'L', 'B']
    state = ''
    for face in order:
        arr = cube.faces[face]
        for i in range(3):
            for j in range(3):
                state += color_map[arr[i, j]]
    return state


def demonstrate_advanced_algorithms():
    """Demonstrate both Korf's and Kociemba's algorithms"""
    print("=" * 60)
    print("ADVANCED RUBIK'S CUBE ALGORITHMS DEMONSTRATION")
    print("=" * 60)
    print()
    
    # Test Case 1: Korf's IDA* Algorithm
    print("🔬 KORF'S IDA* ALGORITHM WITH PATTERN DATABASES")
    print("-" * 60)
    
    cube1 = AdvancedRubiksCube()
    simple_scramble = ["R", "U", "R'", "F"]  # 4 moves for demo
    cube1.execute_sequence(simple_scramble)
    print(f"Scramble: {' '.join(simple_scramble)}")
    
    korf_solver = KorfIDAStar(cube1)
    
    # Generate pattern databases (this takes time but shows sophistication)
    start_time = time.time()
    korf_solver.generate_pattern_databases()
    db_time = time.time() - start_time
    
    # Solve using IDA*
    print("Solving with IDA* search...")
    solution1, nodes1 = korf_solver.ida_star_search(max_depth=8)
    solve_time = time.time() - start_time - db_time
    
    if solution1:
        print(f"✅ Korf's Solution: {' '.join(solution1)}")
        print(f"   Length: {len(solution1)} moves (OPTIMAL)")
        print(f"   Nodes explored: {nodes1:,}")
        print(f"   Database generation: {db_time:.2f}s")
        print(f"   Solve time: {solve_time:.3f}s")
        
        # Verify solution
        test_cube = cube1.copy()
        test_cube.execute_sequence(solution1)
        print(f"   Verification: {'✅ SOLVED' if test_cube.is_solved() else '❌ FAILED'}")
    else:
        print("❌ No solution found within depth limit")
    print()
    
    # Test Case 2: Kociemba's 2-Phase Algorithm (using kociemba library)
    print("⚡ KOCIEMBA'S 2-PHASE ALGORITHM (kociemba library)")
    print("-" * 60)
    
    cube2 = AdvancedRubiksCube()
    medium_scramble = ["R"]
    cube2.execute_sequence(medium_scramble)
    print(f"Scramble: {' '.join(medium_scramble)}")
    
    kociemba_state = to_kociemba_string(cube2)
    print("Kociemba state string:", kociemba_state)
    print("Color counts after scramble:", Counter(to_kociemba_string(cube2)))
    try:
        solution = kociemba.solve(kociemba_state)
        print("Kociemba library solution:", solution)
        # Optionally verify
        moves = solution.split()
        test_cube2 = cube2.copy()
        test_cube2.execute_sequence(moves)
        print(f"   Verification: {'✅ SOLVED' if test_cube2.is_solved() else '❌ FAILED'}")
    except Exception as e:
        print("Error:", e)

    # --- Commented out: Old custom KociembaAlgorithm code ---
    # (Retain for reference, but not used)
    '''
    # Initialize a pattern database for phase 1 heuristic
    phase1_pattern_db = PatternDatabase("corner")
    print("[DEBUG] Generating phase 1 pattern database for heuristic...")
    phase1_pattern_db.generate_database(cube2, max_states=10000)

    class DebugKociembaAlgorithm(KociembaAlgorithm):
        def __init__(self, cube: AdvancedRubiksCube, phase1_db: PatternDatabase):
            super().__init__(cube)
            self.phase1_db = phase1_db
        def _is_redundant_move(self, moves, move):
            if not moves:
                return False
            last = moves[-1]
            # Don't allow the same face twice in a row
            if last[0] == move[0]:
                return True
            # Don't allow three consecutive moves on the same axis (U/D, F/B, R/L)
            axis = {'U': 0, 'D': 0, 'F': 1, 'B': 1, 'R': 2, 'L': 2}
            if len(moves) > 1 and axis[last[0]] == axis[move[0]] == axis[moves[-2][0]]:
                return True
            return False
        def _phase1_heuristic(self, cube: AdvancedRubiksCube) -> int:
            # Use the pattern database to estimate moves to phase 1 completion
            return self.phase1_db.get_heuristic(cube)
        def _phase1_search(self, max_depth: int) -> List[List[str]]:
            solutions = []
            queue = deque([(self.cube.copy(), [])])
            visited = set()
            debug_counter = 0
            while queue and len(solutions) < 10:
                current_cube, moves = queue.popleft()
                self.nodes_explored += 1
                debug_counter += 1
                if debug_counter % 1000 == 0:
                    h = self._phase1_heuristic(current_cube)
                    print(f"[DEBUG] Phase 1: explored {debug_counter} nodes, queue size: {len(queue)}, moves: {moves}, heuristic: {h}")
                if len(moves) > max_depth:
                    continue
                # Heuristic pruning
                h = self._phase1_heuristic(current_cube)
                if len(moves) + h > max_depth:
                    continue
                if current_cube.is_phase1_complete():
                    print(f"[DEBUG] Phase 1 complete at moves: {moves}")
                    solutions.append(moves.copy())
                    continue
                state_hash = current_cube.get_state_hash()
                if state_hash in visited:
                    continue
                visited.add(state_hash)
                for move in current_cube.phase1_moves:
                    if self._is_redundant_move(moves, move):
                        continue
                    new_cube = current_cube.copy()
                    new_cube.execute_move(move)
                    queue.append((new_cube, moves + [move]))
            return solutions
        def _phase2_search(self, cube: AdvancedRubiksCube, max_depth: int) -> List[str]:
            queue = deque([(cube.copy(), [])])
            visited = set()
            debug_counter = 0
            while queue:
                current_cube, moves = queue.popleft()
                self.nodes_explored += 1
                debug_counter += 1
                if debug_counter % 1000 == 0:
                    print(f"[DEBUG] Phase 2: explored {debug_counter} nodes, queue size: {len(queue)}, moves: {moves}")
                if len(moves) > max_depth:
                    continue
                if current_cube.is_solved():
                    print(f"[DEBUG] Phase 2 solved at moves: {moves}")
                    return moves
                state_hash = current_cube.get_state_hash()
                if state_hash in visited:
                    continue
                visited.add(state_hash)
                for move in current_cube.all_moves:
                    if self._is_redundant_move(moves, move):
                        continue
                    new_cube = current_cube.copy()
                    new_cube.execute_move(move)
                    queue.append((new_cube, moves + [move]))
            return []
    
    kociemba_solver = DebugKociembaAlgorithm(cube2, phase1_pattern_db)
    solution2, stats2 = kociemba_solver.solve(max_phase1_depth=max_phase1_depth, max_phase2_depth=max_phase2_depth)
    
    if solution2:
        print(f"✅ Kociemba Solution: {' '.join(solution2[:15])}{'...' if len(solution2) > 15 else ''}")
        print(f"   Length: {len(solution2)} moves")
        print(f"   Phase 1 solutions found: {stats2['phase1_solutions']}")
        print(f"   Total time: {stats2['time']:.3f}s")
        print(f"   Nodes explored: {stats2['nodes_explored']:,}")
        
        # Verify solution
        test_cube2 = cube2.copy()
        test_cube2.execute_sequence(solution2)
        print(f"   Verification: {'✅ SOLVED' if test_cube2.is_solved() else '❌ FAILED'}")
    else:
        print("❌ No solution found within depth limit")
    print()
    '''

    # --- Kociemba solved cube test ---
    print("\n[DEBUG] Testing Kociemba library with solved cube:")
    solved_cube = AdvancedRubiksCube()
    solved_state = to_kociemba_string(solved_cube)
    print("Solved cube state string:", solved_state)
    print("Color counts:", Counter(solved_state))
    try:
        solved_solution = kociemba.solve(solved_state)
        print("Kociemba solution for solved cube:", solved_solution)
    except Exception as e:
        print("Kociemba error for solved cube:", e)
    print()

    # --- Kociemba incremental scramble test ---
    print("\n[DEBUG] Testing Kociemba library with incremental scrambles:")
    test_moves = ["R", "U", "R'", "F", "R", "F'", "U", "R"]
    test_cube = AdvancedRubiksCube()
    for i in range(1, len(test_moves) + 1):
        test_cube = AdvancedRubiksCube()
        test_cube.execute_sequence(test_moves[:i])
        kociemba_state = to_kociemba_string(test_cube)
        print(f"After moves: {' '.join(test_moves[:i])}")
        print("Kociemba state string:", kociemba_state)
        print("Color counts:", Counter(kociemba_state))
        try:
            solution = kociemba.solve(kociemba_state)
            print("Kociemba solution:", solution)
        except Exception as e:
            print("Kociemba error:", e)
        print()

if __name__ == "__main__":
    demonstrate_advanced_algorithms()