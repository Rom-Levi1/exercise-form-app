// src/App.js
import React, { useState } from "react";
import "./App.css";

import HomeScreen from "./Homepage/Homepage";
import AuthPage from "./AuthPage/AuthPage";

function App() {
  const [showAuth, setShowAuth] = useState(false);

  return showAuth ? (
    <AuthPage onBackClick={() => setShowAuth(false)} />
  ) : (
    <HomeScreen onLoginClick={() => setShowAuth(true)} />
  );
}

export default App;