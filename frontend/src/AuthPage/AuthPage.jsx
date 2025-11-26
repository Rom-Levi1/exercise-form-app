// src/AuthPage.jsx
import { useState } from "react";
import "./AuthPage.css";

function AuthPage({ onBackClick }) {
  const [mode, setMode] = useState("login"); // "login" | "register"

  const handleSubmit = (e) => {
    e.preventDefault();
    // later this will call the backend
    console.log(`${mode} form submitted`);
  };

  return (
    <div className="auth-page">
      {/* Top bar – same vibe as the main page */}
      <header className="auth-header">
        <div
          className="logo clickable-logo"
          onClick={onBackClick}
        >
          Tech-nique
        </div>
      </header>


      <main className="auth-main">
        <div className="auth-card">
          {/* Tabs */}
          <div className="auth-tabs">
            <button
              className={mode === "login" ? "tab active" : "tab"}
              onClick={() => setMode("login")}
              type="button"
            >
              Login
            </button>
            <button
              className={mode === "register" ? "tab active" : "tab"}
              onClick={() => setMode("register")}
              type="button"
            >
              Register
            </button>
          </div>

          {/* Title + description */}
          <h1 className="auth-title">
            {mode === "login"
              ? "Log in to Tech-nique"
              : "Create your Tech-nique account"}
          </h1>
          <p className="auth-subtitle">
            {mode === "login"
              ? "Access your saved videos and form analysis."
              : "Save your workouts and track form improvements over time."}
          </p>

          {/* Form */}
          <form onSubmit={handleSubmit} className="auth-form">
            <label className="auth-label">
              Email
              <input
                type="email"
                className="auth-input"
                placeholder="you@example.com"
                required
              />
            </label>

            <label className="auth-label">
              Password
              <input
                type="password"
                className="auth-input"
                placeholder="••••••••"
                required
              />
            </label>

            {mode === "register" && (
              <label className="auth-label">
                Confirm password
                <input
                  type="password"
                  className="auth-input"
                  placeholder="Repeat password"
                  required
                />
              </label>
            )}

            <button className="auth-primary-btn" type="submit" disabled>
              {mode === "login" ? "Login" : "Create account"}
            </button>

            <p className="auth-note">
              * Button is disabled for now – backend connection will be added
              later.
            </p>
          </form>
        </div>
      </main>

      <footer className="auth-footer">
        © {new Date().getFullYear()} Tech-nique · Created by Ofek Daniel &amp;
        Rom Levi
      </footer>
    </div>
  );
}

export default AuthPage;
