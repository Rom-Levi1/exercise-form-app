// src/App.js
import React from "react";
import { Routes, Route, useNavigate } from "react-router-dom";

import HomePage from "./Homepage/Homepage.jsx";
import AuthPage from "./AuthPage/AuthPage.jsx";
import UploadPage from "./UploadPage/UploadPage.jsx";
import ResultPage from "./ResultPage/ResultPage.jsx";

function App() {
  const navigate = useNavigate();

  return (
    <Routes>
      {/* Home */}
      <Route
        path="/"
        element={
          <HomePage
            onLoginClick={() => navigate("/auth")}
            onLogoClick={() => navigate("/")}
            onUploadClick={() => navigate("/upload")} // 👈 new
          />
        }
      />

      {/* Auth (even if not fully implemented yet) */}
      <Route
        path="/auth"
        element={<AuthPage onLogoClick={() => navigate("/")} />}
      />

      {/* Upload – our new empty page */}
      <Route
        path="/upload"
        element={<UploadPage onLogoClick={() => navigate("/")} />}
      />

      <Route
        path="/result"
        element={
          <ResultPage
            onLogoClick={() => navigate("/")}
            onBackToUpload={() => navigate("/upload")}
          />
        }
      />
    </Routes>
  );
}

export default App;
