import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from flask import Flask, request, jsonify
from cubesolver.cub3 import AdvancedRubiksCube, KorfIDAStar, BeginnerSolver
from flask_cors import CORS

app = Flask(__name__)
CORS(app)  # Allow requests from frontend

cube = AdvancedRubiksCube()

def serialize_cube_faces(faces):
    # Convert numpy arrays to lists for JSON serialization
    return {face: arr.tolist() for face, arr in faces.items()}

@app.route('/scramble', methods=['POST'])
def scramble():
    moves = cube.scramble()
    return jsonify({'moves': moves, 'state': serialize_cube_faces(cube.faces)})

@app.route('/solve', methods=['POST'])
def solve():
    solver = KorfIDAStar(cube)
    solver.generate_pattern_databases()
    solution, nodes = solver.ida_star_search(max_depth=8)
    
    # Apply the solution to get the solved state
    solved_cube = cube.copy()
    solved_cube.execute_sequence(solution)
    
    return jsonify({
        'solution': solution, 
        'nodes': nodes,
        'solved_state': serialize_cube_faces(solved_cube.faces)
    })

@app.route('/beginner_solve', methods=['POST'])
def beginner_solve():
    """Solve using the beginner's method (fast, always works)"""
    solver = BeginnerSolver(cube)
    solution = solver.solve()
    
    # Apply the solution to get the solved state
    solved_cube = cube.copy()
    solved_cube.execute_sequence(solution)
    
    return jsonify({
        'solution': solution, 
        'method': 'beginner',
        'solved_state': serialize_cube_faces(solved_cube.faces)
    })

@app.route('/state', methods=['GET'])
def state():
    return jsonify({'state': serialize_cube_faces(cube.faces)})

@app.route('/reset', methods=['POST'])
def reset():
    global cube
    cube = AdvancedRubiksCube()
    return jsonify({'state': serialize_cube_faces(cube.faces)})

if __name__ == '__main__':
    app.run(debug=True)