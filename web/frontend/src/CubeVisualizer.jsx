import React from 'react';

const colorMap = ['#fff', '#f00', '#00f', '#ffa500', '#0f0', '#ff0']; // W, R, B, O, G, Y
const faceLabels = { 'U': 'Up', 'D': 'Down', 'F': 'Front', 'B': 'Back', 'L': 'Left', 'R': 'Right' };

function CubeFace({ face, data }) {
  return (
    <div style={{ textAlign: 'center', margin: '5px' }}>
      <div style={{ 
        fontSize: '12px', 
        fontWeight: 'bold', 
        marginBottom: '5px',
        color: '#666'
      }}>
        {faceLabels[face]}
      </div>
      <div style={{ 
        display: 'grid', 
        gridTemplateColumns: 'repeat(3, 25px)', 
        gap: 2,
        border: '2px solid #333',
        borderRadius: '5px',
        padding: '3px',
        background: '#f0f0f0'
      }}>
        {data.flat().map((color, i) => (
          <div
            key={i}
            style={{
              width: 25,
              height: 25,
              background: colorMap[color],
              border: '1px solid #333',
              borderRadius: '3px',
              boxShadow: 'inset 0 0 3px rgba(0,0,0,0.3)'
            }}
          />
        ))}
      </div>
    </div>
  );
}

export default function CubeVisualizer({ state }) {
  if (!state) return (
    <div style={{ 
      textAlign: 'center', 
      color: '#666', 
      fontSize: '16px',
      padding: '20px'
    }}>
      No cube state available
    </div>
  );
  
  return (
    <div style={{ 
      display: 'flex', 
      flexDirection: 'column', 
      alignItems: 'center',
      background: '#fafafa',
      padding: '20px',
      borderRadius: '10px',
      border: '1px solid #ddd'
    }}>
      <h3 style={{ 
        margin: '0 0 15px 0', 
        color: '#333',
        fontSize: '18px'
      }}>
        🎯 Cube State
      </h3>
      
      {/* 2D Net Layout */}
      <div style={{ display: 'flex', flexDirection: 'column', alignItems: 'center' }}>
        {/* Top face */}
        <div style={{ marginBottom: 10 }}>
          <CubeFace face="U" data={state.U} />
        </div>
        
        {/* Middle row: Left, Front, Right, Back */}
        <div style={{ display: 'flex', gap: 5 }}>
          <CubeFace face="L" data={state.L} />
          <CubeFace face="F" data={state.F} />
          <CubeFace face="R" data={state.R} />
          <CubeFace face="B" data={state.B} />
        </div>
        
        {/* Bottom face */}
        <div style={{ marginTop: 10 }}>
          <CubeFace face="D" data={state.D} />
        </div>
      </div>
      
      {/* Color Legend */}
      <div style={{ 
        marginTop: '15px', 
        fontSize: '12px', 
        color: '#666',
        textAlign: 'center'
      }}>
        <strong>Colors:</strong> White, Red, Blue, Orange, Green, Yellow
      </div>
    </div>
  );
}