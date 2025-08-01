# 🎯 Rubik's Cube Solver Web Application

A complete web application that demonstrates advanced Rubik's Cube solving algorithms with an interactive frontend.

## 🚀 Features

- **Two Solving Algorithms:**
  - **IDA* (Optimal):** Finds shortest solutions using pattern databases
  - **Beginner Method (Fast):** Always works, reasonable solution lengths
- **Interactive Web Interface:** Real-time cube visualization and control
- **RESTful API:** Flask backend with JSON endpoints
- **Modern Frontend:** React with Vite for fast development

## 🏗️ Architecture

```
aerohack/
├── cubesolver/          # Python cube logic and algorithms
│   ├── cub3.py         # Main cube implementation with solvers
│   ├── cube.py         # Basic cube implementation
│   └── cube2.py        # Alternative implementation
├── web/                # Web application
│   ├── app.py          # Flask backend API
│   └── frontend/       # React frontend
│       ├── src/
│       │   ├── App.jsx           # Main application component
│       │   └── CubeVisualizer.jsx # Cube visualization
│       └── package.json
├── rubikscubesolver/   # TypeScript cube implementation (reference)
└── test_web_integration.py  # API testing script
```

## 🛠️ Installation & Setup

### Prerequisites
- Python 3.8+
- Node.js 16+
- npm or yarn

### Backend Setup
```bash
# Install Python dependencies
pip install flask flask-cors numpy kociemba

# Start the Flask backend
cd web
python app.py
```

### Frontend Setup
```bash
# Install frontend dependencies
cd web/frontend
npm install

# Start the development server
npm run dev
```

## 🌐 Usage

1. **Start the backend:**
   ```bash
   cd web
   python app.py
   ```
   Backend will run on `http://localhost:5000`

2. **Start the frontend:**
   ```bash
   cd web/frontend
   npm run dev
   ```
   Frontend will run on `http://localhost:5173`

3. **Open your browser** to `http://localhost:5173`

## 🎮 How to Use

### Web Interface
- **🎲 Scramble:** Randomize the cube with 25 random moves
- **🧠 Solve (IDA*):** Find optimal solution (slower, shorter moves)
- **⚡ Solve (Beginner):** Find fast solution (always works)
- **🔄 Reset:** Return to solved state

### API Endpoints
- `GET /state` - Get current cube state
- `POST /scramble` - Scramble the cube
- `POST /solve` - Solve with IDA* algorithm
- `POST /beginner_solve` - Solve with beginner method
- `POST /reset` - Reset to solved state

## 🔬 Algorithms

### IDA* (Iterative Deepening A*)
- Uses pattern databases for heuristics
- Finds optimal solutions
- Slower for large scrambles
- Best for research and optimal solving

### Beginner Method
- Uses Kociemba library as backend
- Always finds a solution quickly
- Reasonable solution lengths (~16 moves)
- Perfect for web demos and user experience

## 📊 Performance

| Algorithm | Solution Length | Speed | Use Case |
|-----------|----------------|-------|----------|
| IDA* | ~11 moves | Slow (1-2s) | Optimal solving |
| Beginner | ~16 moves | Fast (<100ms) | Web demos |

## 🧪 Testing

Run the API test script:
```bash
python test_web_integration.py
```

This will test all endpoints and verify the system is working correctly.

## 🎨 Features

### Cube Visualization
- 2D net representation of all 6 faces
- Color-coded squares (White, Red, Blue, Orange, Green, Yellow)
- Real-time state updates
- Face labels for clarity

### User Experience
- Loading states during operations
- Error handling and display
- Responsive design
- Clear instructions and feedback

## 🔧 Technical Details

### Backend (Python/Flask)
- **AdvancedRubiksCube:** Full cube implementation with move support
- **KorfIDAStar:** Optimal solving with pattern databases
- **BeginnerSolver:** Fast solving using Kociemba library
- **RESTful API:** JSON endpoints with CORS support

### Frontend (React/Vite)
- **Modern React:** Hooks and functional components
- **Vite:** Fast development and building
- **Interactive UI:** Real-time cube visualization
- **Error Handling:** User-friendly error messages

## 🚀 Deployment

### Development
```bash
# Terminal 1: Backend
cd web && python app.py

# Terminal 2: Frontend
cd web/frontend && npm run dev
```

### Production
```bash
# Build frontend
cd web/frontend && npm run build

# Serve with production WSGI server
cd web && gunicorn app:app
```

## 📝 API Response Format

### State Response
```json
{
  "state": {
    "U": [[0,0,0], [0,0,0], [0,0,0]],
    "D": [[5,5,5], [5,5,5], [5,5,5]],
    "F": [[2,2,2], [2,2,2], [2,2,2]],
    "B": [[3,3,3], [3,3,3], [3,3,3]],
    "L": [[4,4,4], [4,4,4], [4,4,4]],
    "R": [[1,1,1], [1,1,1], [1,1,1]]
  }
}
```

### Solve Response
```json
{
  "solution": ["R", "U", "R'", "U'"],
  "method": "beginner"
}
```

## 🎯 Future Enhancements

- [ ] 3D cube visualization with Three.js
- [ ] Step-by-step solution animation
- [ ] Custom scramble input
- [ ] Solution optimization options
- [ ] Mobile-responsive design
- [ ] User accounts and solve history

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Test thoroughly
5. Submit a pull request

## 📄 License

This project is open source and available under the MIT License.

---

**🎉 Enjoy solving Rubik's Cubes with advanced algorithms!** 