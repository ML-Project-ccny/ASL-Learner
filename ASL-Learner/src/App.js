import React from 'react';
import './App.css';
import Home from './home';
import {Routes, Route} from "react-router-dom";
import 'bootstrap/dist/css/bootstrap.min.css';
import ChooseLevel from './chooseLevel'
import Game from './game'
import Login from './Login';
import Register from './Register';
import Info from './info';
import { useState } from 'react';

function App() {
  const [email, setUser] = useState('');
  const [password, setPwd] = useState('');

  return (
    <div className="App">
        <style jsx global>{`
        body {
          margin: 0px;
          padding: 0px;
        }
      `}</style>
      <Routes>
        <Route path="/" element={<Home/>} />
        <Route path="/level" element={<ChooseLevel/>} />
        <Route path="/game" element={<Game />} />
        <Route path="/Login" element={<Login email={email} setUser={setUser} password={password} setPwd={setPwd}/>} />
        <Route path="/Register" element={<Register email={email} setUser={setUser} password={password} setPwd={setPwd}/>} />
        <Route path="/Info" element={<Info/>} />

      </Routes>
    </div>
  );
}

export default App;
