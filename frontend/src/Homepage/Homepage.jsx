// src/screens/HomeScreen.js
import React from "react";
import "../App.css";
import "./Homepage.css";

function HomeScreen({ onLoginClick }) {
  return (
    <div className="app">
      {/* Top bar */}
      <header className="app-header">
        <div className="logo">Tech-nique</div>
        <nav className="nav">
          <a href="#how-it-works">How it works</a>
          <a href="#features">Features</a>
          <a href="#about">About</a>
        </nav>
      </header>

      {/* Hero / Login section */}
      <main className="hero">
        <div className="hero-text">
          <h1>Let’s Start</h1>
          <p>Log in to upload your workout videos and get AI-powered form analysis.</p>

          {/* Login button triggers parent callback */}
          <button className="primary-btn" onClick={onLoginClick}>
            Login
          </button>

          <p className="hero-note">Ready when you are 💪</p>
        </div>
      </main>

      {/* How it works */}
      <section id="how-it-works" className="section">
        <h2>How it works</h2>
        <div className="steps">
          <div className="step-card">
            <h3>1. Login</h3>
            <p>You need to log in to upload videos and see your analysis.</p>
          </div>
          <div className="step-card">
            <h3>2. Upload</h3>
            <p>Choose a pre-recorded workout video from any device.</p>
          </div>
          <div className="step-card">
            <h3>3. Analyze</h3>
            <p>
              Our engine (OpenCV + MediaPipe) detects key joints and calculates
              angles.
            </p>
          </div>
          <div className="step-card">
            <h3>4. Improve</h3>
            <p>Get feedback like “knee too far forward” or “squat depth too shallow”.</p>
          </div>
        </div>
      </section>

      {/* Features */}
      <section id="features" className="section section-alt">
        <h2>Main Features</h2>
        <ul className="feature-list">
          <li>Angle calculation for hips, knees, and ankles</li>
          <li>Works with pre-recorded videos (no live streaming needed)</li>
          <li>Clear text feedback and basic scoring</li>
          <li>Helps people training alone at the gym or home</li>
        </ul>
      </section>

      {/* About */}
      <section id="about" className="section">
        <h2>About this project</h2>
        <p>
          This app combines workout knowledge with computer vision. It starts with form analysis
          for basic lifts and can later grow into a platform with history, progress tracking, and more.
        </p>
      </section>

      <footer className="footer">
        <p>© {new Date().getFullYear()} Tech-nique · Built with React</p>
      </footer>
    </div>
  );
}

export default HomeScreen;
