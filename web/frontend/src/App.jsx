import React, { useState, useEffect } from 'react';
import CubeVisualizer from './CubeVisualizer';

function App() {
  const [cubeState, setCubeState] = useState(null);
  const [solution, setSolution] = useState([]);
  const [scrambleMoves, setScrambleMoves] = useState([]);
  const [solveMethod, setSolveMethod] = useState('');
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState('');

  // Fetch initial state
  useEffect(() => {
    fetchInitialState();
  }, []);

  const fetchInitialState = async () => {
    try {
      setLoading(true);
      const response = await fetch('http://localhost:5000/state');
      if (response.ok) {
        const data = await response.json();
        setCubeState(data.state);
        setError('');
      } else {
        setError('Failed to fetch cube state');
      }
    } catch (err) {
      setError('Error connecting to server');
    } finally {
      setLoading(false);
    }
  };

  const scramble = async () => {
    try {
      setLoading(true);
      setError('');
      const response = await fetch('http://localhost:5000/scramble', { method: 'POST' });
      if (response.ok) {
        const data = await response.json();
        setCubeState(data.state);
        setScrambleMoves(data.moves);
        setSolution([]);
        setSolveMethod('');
      } else {
        setError('Failed to scramble cube');
      }
    } catch (err) {
      setError('Error scrambling cube');
    } finally {
      setLoading(false);
    }
  };

  const solve = async () => {
    try {
      setLoading(true);
      setError('');
      const response = await fetch('http://localhost:5000/solve', { method: 'POST' });
      if (response.ok) {
        const data = await response.json();
        setSolution(data.solution);
        setSolveMethod('IDA* (Optimal)');
        
        // Use the solved state from the backend
        if (data.solved_state) {
          console.log('Setting solved state from backend:', data.solved_state);
          setCubeState(data.solved_state);
        } else {
          // Fallback: try to apply moves (for backward compatibility)
          await applySolutionToCube(data.solution);
        }
      } else {
        setError('Failed to solve with IDA*');
      }
    } catch (err) {
      setError('Error solving with IDA*');
    } finally {
      setLoading(false);
    }
  };

  const beginnerSolve = async () => {
    try {
      setLoading(true);
      setError('');
      const response = await fetch('http://localhost:5000/beginner_solve', { method: 'POST' });
      if (response.ok) {
        const data = await response.json();
        setSolution(data.solution);
        setSolveMethod('Beginner Method (Fast)');
        
        // Use the solved state from the backend
        if (data.solved_state) {
          console.log('Setting solved state from backend:', data.solved_state);
          setCubeState(data.solved_state);
        } else {
          // Fallback: try to apply moves (for backward compatibility)
          await applySolutionToCube(data.solution);
        }
      } else {
        setError('Failed to solve with beginner method');
      }
    } catch (err) {
      setError('Error solving with beginner method');
    } finally {
      setLoading(false);
    }
  };

  const applySolutionToCube = async (solutionMoves) => {
    try {
      console.log('Applying solution moves:', solutionMoves);
      console.log('Current cube state before applying moves:', cubeState);
      
      // Apply each move to the cube state
      let currentState = { ...cubeState };
      
      for (const move of solutionMoves) {
        console.log('Applying move:', move);
        // Apply the move to the current state
        currentState = applyMoveToState(currentState, move);
        console.log('State after move:', currentState);
      }
      
      console.log('Final state after applying all moves:', currentState);
      
      // Update the cube state with the solved state
      setCubeState(currentState);
    } catch (err) {
      console.error('Error applying solution:', err);
    }
  };

  const applyMoveToState = (state, move) => {
    // Create a deep copy of the state
    const newState = {
      U: state.U.map(row => [...row]),
      D: state.D.map(row => [...row]),
      F: state.F.map(row => [...row]),
      B: state.B.map(row => [...row]),
      L: state.L.map(row => [...row]),
      R: state.R.map(row => [...row])
    };

    // Apply the move based on the move notation
    if (move === 'U') {
      // Rotate U face clockwise
      newState.U = rotateFaceClockwise(newState.U);
      // Rotate adjacent edges
      const temp = [...newState.F[0]];
      newState.F[0] = [...newState.L[0]];
      newState.L[0] = [...newState.B[0]];
      newState.B[0] = [...newState.R[0]];
      newState.R[0] = temp;
    } else if (move === "U'") {
      // Rotate U face counter-clockwise
      newState.U = rotateFaceCounterClockwise(newState.U);
      // Rotate adjacent edges
      const temp = [...newState.F[0]];
      newState.F[0] = [...newState.R[0]];
      newState.R[0] = [...newState.B[0]];
      newState.B[0] = [...newState.L[0]];
      newState.L[0] = temp;
    } else if (move === 'U2') {
      // Rotate U face 180 degrees
      newState.U = rotateFaceClockwise(rotateFaceClockwise(newState.U));
      // Rotate adjacent edges
      const temp = [...newState.F[0]];
      newState.F[0] = [...newState.B[0]];
      newState.B[0] = temp;
      const temp2 = [...newState.L[0]];
      newState.L[0] = [...newState.R[0]];
      newState.R[0] = temp2;
    } else if (move === 'R') {
      // Rotate R face clockwise
      newState.R = rotateFaceClockwise(newState.R);
      // Rotate adjacent edges
      const temp = [newState.U[0][2], newState.U[1][2], newState.U[2][2]];
      newState.U[0][2] = newState.F[0][2];
      newState.U[1][2] = newState.F[1][2];
      newState.U[2][2] = newState.F[2][2];
      newState.F[0][2] = newState.D[0][2];
      newState.F[1][2] = newState.D[1][2];
      newState.F[2][2] = newState.D[2][2];
      newState.D[0][2] = newState.B[2][0];
      newState.D[1][2] = newState.B[1][0];
      newState.D[2][2] = newState.B[0][0];
      newState.B[2][0] = temp[0];
      newState.B[1][0] = temp[1];
      newState.B[0][0] = temp[2];
    } else if (move === "R'") {
      // Rotate R face counter-clockwise
      newState.R = rotateFaceCounterClockwise(newState.R);
      // Rotate adjacent edges
      const temp = [newState.U[0][2], newState.U[1][2], newState.U[2][2]];
      newState.U[0][2] = newState.B[2][0];
      newState.U[1][2] = newState.B[1][0];
      newState.U[2][2] = newState.B[0][0];
      newState.B[2][0] = newState.D[0][2];
      newState.B[1][0] = newState.D[1][2];
      newState.B[0][0] = newState.D[2][2];
      newState.D[0][2] = newState.F[0][2];
      newState.D[1][2] = newState.F[1][2];
      newState.D[2][2] = newState.F[2][2];
      newState.F[0][2] = temp[0];
      newState.F[1][2] = temp[1];
      newState.F[2][2] = temp[2];
    } else if (move === 'F') {
      // Rotate F face clockwise
      newState.F = rotateFaceClockwise(newState.F);
      // Rotate adjacent edges
      const temp = [...newState.U[2]];
      newState.U[2] = [newState.L[2][2], newState.L[1][2], newState.L[0][2]];
      newState.L[2][2] = newState.D[0][2];
      newState.L[1][2] = newState.D[0][1];
      newState.L[0][2] = newState.D[0][0];
      newState.D[0] = [newState.R[0][0], newState.R[1][0], newState.R[2][0]];
      newState.R[0][0] = temp[2];
      newState.R[1][0] = temp[1];
      newState.R[2][0] = temp[0];
    } else if (move === "F'") {
      // Rotate F face counter-clockwise
      newState.F = rotateFaceCounterClockwise(newState.F);
      // Rotate adjacent edges
      const temp = [...newState.U[2]];
      newState.U[2] = [newState.R[2][0], newState.R[1][0], newState.R[0][0]];
      newState.R[2][0] = newState.D[0][0];
      newState.R[1][0] = newState.D[0][1];
      newState.R[0][0] = newState.D[0][2];
      newState.D[0] = [newState.L[0][2], newState.L[1][2], newState.L[2][2]];
      newState.L[0][2] = temp[2];
      newState.L[1][2] = temp[1];
      newState.L[2][2] = temp[0];
    } else if (move === 'D') {
      // Rotate D face clockwise
      newState.D = rotateFaceClockwise(newState.D);
      // Rotate adjacent edges
      const temp = [...newState.F[2]];
      newState.F[2] = [...newState.R[2]];
      newState.R[2] = [...newState.B[2]];
      newState.B[2] = [...newState.L[2]];
      newState.L[2] = temp;
    } else if (move === "D'") {
      // Rotate D face counter-clockwise
      newState.D = rotateFaceCounterClockwise(newState.D);
      // Rotate adjacent edges
      const temp = [...newState.F[2]];
      newState.F[2] = [...newState.L[2]];
      newState.L[2] = [...newState.B[2]];
      newState.B[2] = [...newState.R[2]];
      newState.R[2] = temp;
    } else if (move === 'L') {
      // Rotate L face clockwise
      newState.L = rotateFaceClockwise(newState.L);
      // Rotate adjacent edges
      const temp = [newState.U[0][0], newState.U[1][0], newState.U[2][0]];
      newState.U[0][0] = newState.B[2][2];
      newState.U[1][0] = newState.B[1][2];
      newState.U[2][0] = newState.B[0][2];
      newState.B[2][2] = newState.D[0][0];
      newState.B[1][2] = newState.D[1][0];
      newState.B[0][2] = newState.D[2][0];
      newState.D[0][0] = newState.F[0][0];
      newState.D[1][0] = newState.F[1][0];
      newState.D[2][0] = newState.F[2][0];
      newState.F[0][0] = temp[0];
      newState.F[1][0] = temp[1];
      newState.F[2][0] = temp[2];
    } else if (move === "L'") {
      // Rotate L face counter-clockwise
      newState.L = rotateFaceCounterClockwise(newState.L);
      // Rotate adjacent edges
      const temp = [newState.U[0][0], newState.U[1][0], newState.U[2][0]];
      newState.U[0][0] = newState.F[0][0];
      newState.U[1][0] = newState.F[1][0];
      newState.U[2][0] = newState.F[2][0];
      newState.F[0][0] = newState.D[0][0];
      newState.F[1][0] = newState.D[1][0];
      newState.F[2][0] = newState.D[2][0];
      newState.D[0][0] = newState.B[2][2];
      newState.D[1][0] = newState.B[1][2];
      newState.D[2][0] = newState.B[0][2];
      newState.B[2][2] = temp[0];
      newState.B[1][2] = temp[1];
      newState.B[0][2] = temp[2];
    } else if (move === 'B') {
      // Rotate B face clockwise
      newState.B = rotateFaceClockwise(newState.B);
      // Rotate adjacent edges
      const temp = [...newState.U[0]];
      newState.U[0] = [newState.R[0][2], newState.R[1][2], newState.R[2][2]];
      newState.R[0][2] = newState.D[2][2];
      newState.R[1][2] = newState.D[2][1];
      newState.R[2][2] = newState.D[2][0];
      newState.D[2] = [newState.L[0][0], newState.L[1][0], newState.L[2][0]];
      newState.L[0][0] = temp[2];
      newState.L[1][0] = temp[1];
      newState.L[2][0] = temp[0];
    } else if (move === "B'") {
      // Rotate B face counter-clockwise
      newState.B = rotateFaceCounterClockwise(newState.B);
      // Rotate adjacent edges
      const temp = [...newState.U[0]];
      newState.U[0] = [newState.L[2][0], newState.L[1][0], newState.L[0][0]];
      newState.L[2][0] = newState.D[2][0];
      newState.L[1][0] = newState.D[2][1];
      newState.L[0][0] = newState.D[2][2];
      newState.D[2] = [newState.R[2][2], newState.R[1][2], newState.R[0][2]];
      newState.R[2][2] = temp[2];
      newState.R[1][2] = temp[1];
      newState.R[0][2] = temp[0];
    }

    return newState;
  };

  const rotateFaceClockwise = (face) => {
    return [
      [face[2][0], face[1][0], face[0][0]],
      [face[2][1], face[1][1], face[0][1]],
      [face[2][2], face[1][2], face[0][2]]
    ];
  };

  const rotateFaceCounterClockwise = (face) => {
    return [
      [face[0][2], face[1][2], face[2][2]],
      [face[0][1], face[1][1], face[2][1]],
      [face[0][0], face[1][0], face[2][0]]
    ];
  };

  const reset = async () => {
    try {
      setLoading(true);
      setError('');
      const response = await fetch('http://localhost:5000/reset', { method: 'POST' });
      if (response.ok) {
        const data = await response.json();
        setCubeState(data.state);
        setScrambleMoves([]);
        setSolution([]);
        setSolveMethod('');
      } else {
        setError('Failed to reset cube');
      }
    } catch (err) {
      setError('Error resetting cube');
    } finally {
      setLoading(false);
    }
  };

  return (
    <div style={{ padding: 20, maxWidth: 800, margin: '0 auto' }}>
      <h1 style={{ textAlign: 'center', color: '#333', marginBottom: 30 }}>
        🎯 Rubik's Cube Solver Demo
      </h1>
      
      {/* Error Display */}
      {error && (
        <div style={{ 
          background: '#ffebee', 
          color: '#c62828', 
          padding: 10, 
          borderRadius: 5, 
          marginBottom: 20,
          textAlign: 'center'
        }}>
          ❌ {error}
        </div>
      )}

      {/* Control Buttons */}
      <div style={{ 
        display: 'flex', 
        justifyContent: 'center', 
        gap: 10, 
        marginBottom: 30,
        flexWrap: 'wrap'
      }}>
        <button 
          onClick={scramble} 
          disabled={loading}
          style={{
            padding: '10px 20px',
            fontSize: '16px',
            backgroundColor: '#4CAF50',
            color: 'white',
            border: 'none',
            borderRadius: '5px',
            cursor: loading ? 'not-allowed' : 'pointer',
            opacity: loading ? 0.6 : 1
          }}
        >
          {loading ? '⏳ Scrambling...' : '🎲 Scramble'}
        </button>
        
        <button 
          onClick={solve} 
          disabled={loading}
          style={{
            padding: '10px 20px',
            fontSize: '16px',
            backgroundColor: '#2196F3',
            color: 'white',
            border: 'none',
            borderRadius: '5px',
            cursor: loading ? 'not-allowed' : 'pointer',
            opacity: loading ? 0.6 : 1
          }}
        >
          {loading ? '⏳ Solving...' : '🧠 Solve (IDA*)'}
        </button>
        
        <button 
          onClick={beginnerSolve} 
          disabled={loading}
          style={{
            padding: '10px 20px',
            fontSize: '16px',
            backgroundColor: '#FF9800',
            color: 'white',
            border: 'none',
            borderRadius: '5px',
            cursor: loading ? 'not-allowed' : 'pointer',
            opacity: loading ? 0.6 : 1
          }}
        >
          {loading ? '⏳ Solving...' : '⚡ Solve (Beginner)'}
        </button>
        
        <button 
          onClick={reset} 
          disabled={loading}
          style={{
            padding: '10px 20px',
            fontSize: '16px',
            backgroundColor: '#9C27B0',
            color: 'white',
            border: 'none',
            borderRadius: '5px',
            cursor: loading ? 'not-allowed' : 'pointer',
            opacity: loading ? 0.6 : 1
          }}
        >
          🔄 Reset
        </button>
      </div>

      {/* Cube Visualization */}
      <div style={{ 
        display: 'flex', 
        justifyContent: 'center', 
        marginBottom: 30,
        minHeight: 200
      }}>
        {loading ? (
          <div style={{ 
            display: 'flex', 
            alignItems: 'center', 
            justifyContent: 'center',
            fontSize: '18px',
            color: '#666'
          }}>
            ⏳ Loading cube state...
          </div>
        ) : (
          <CubeVisualizer state={cubeState} />
        )}
      </div>

      {/* Information Display */}
      <div style={{ 
        background: '#f5f5f5', 
        padding: 20, 
        borderRadius: 10,
        marginBottom: 20
      }}>
        {scrambleMoves.length > 0 && (
          <div style={{ marginBottom: 15 }}>
            <strong>🎲 Scramble:</strong> 
            <span style={{ 
              fontFamily: 'monospace', 
              fontSize: '14px',
              marginLeft: 10,
              color: '#333'
            }}>
              {scrambleMoves.join(' ')}
            </span>
          </div>
        )}
        
        {solution.length > 0 && (
          <div>
            <strong>✅ Solution ({solveMethod}):</strong>
            <div style={{ 
              fontFamily: 'monospace', 
              fontSize: '14px',
              marginTop: 5,
              color: '#333',
              wordBreak: 'break-word'
            }}>
              {solution.join(' ')}
            </div>
            <div style={{ 
              marginTop: 10, 
              fontSize: '14px', 
              color: '#666' 
            }}>
              📊 Length: {solution.length} moves
            </div>
          </div>
        )}
      </div>

      {/* Instructions */}
      <div style={{ 
        background: '#e3f2fd', 
        padding: 15, 
        borderRadius: 8,
        fontSize: '14px',
        color: '#1565c0'
      }}>
        <strong>💡 Instructions:</strong>
        <ul style={{ margin: '10px 0 0 20px' }}>
          <li>Click "Scramble" to randomize the cube</li>
          <li>Click "Solve (IDA*)" for optimal solutions (slower but shorter)</li>
          <li>Click "Solve (Beginner)" for fast solutions (always works)</li>
          <li>Click "Reset" to return to solved state</li>
          <li><strong>Note:</strong> The cube visualization will update to show the solved state!</li>
        </ul>
      </div>
    </div>
  );
}

export default App;