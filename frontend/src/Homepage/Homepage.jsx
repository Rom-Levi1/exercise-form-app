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
          <h1>Let’s Start Improving Your Form</h1>
          <p>
            Train smarter, even when you're training alone. Upload your workout
            videos and get precise feedback on angles, depth, and alignment using
            AI pose detection and computer vision.
          </p>

          <p className="hero-extra">
            Tech-nique helps you lift safer, move better, and track progress over time —
            without needing a coach next to you for every session.
          </p>

      {/* Login button */}
      <button className="primary-btn login-wide" onClick={onLoginClick}>
        Login to Start Your Progress
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
            <p>Our engine detects key joints and calculates angles for hips, knees, and ankles using OpenCV and MediaPipe.</p>
          </div>

          <div className="step-card">
            <h3>4. Improve</h3>
            <p>Get actionable feedback like “knee too far forward” or “squat depth too shallow”, and track progress over time.</p>
          </div>
        </div>
      </section>

      {/* Features */}
      <section id="features" className="section section-alt">
        <h2>Main Features</h2>
        <ul className="feature-list">
          <li>Angle calculation for hips, knees, and ankles</li>
          <li>Works with pre-recorded videos (no live streaming needed)</li>
          <li>Clear text feedback and basic scoring for technique</li>
          <li>Designed for people training alone at the gym or at home</li>
        </ul>
      </section>

      {/* About */}
      <section id="about" className="section">
        <h2>About this project</h2>
        <p>
          Tech-nique combines workout knowledge with computer vision. It starts
          with form analysis for basic lifts like squats and deadlifts, and can
          later grow into a full platform with history, trends, and progress
          tracking for your training.
        </p>
      </section>

      <footer className="footer">
        <p>© {new Date().getFullYear()} Tech-nique · Created by Rom Levi &amp;
        Ofek Daniel</p>
      </footer>
    </div>
  );
}

export default HomeScreen;
