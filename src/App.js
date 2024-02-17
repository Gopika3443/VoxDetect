import React from "react";
import { BrowserRouter as Router, Route, Routes } from 'react-router-dom';
import Front from './components/Front';
function App() {
  return(
    <Router>
      <Routes>
        <Route path='/' element={<Front/>}></Route>
      </Routes>
    </Router>

  );
}

export default App;
