// src/UploadPage/UploadPage.jsx
import React from "react";
import "../App.css";
import "./UploadPage.css";

function UploadPage({ onLogoClick }) {
  return (
    <div className="app upload-page">
      {/* Top bar with logo only */}
      <header className="app-header">
        <div className="logo clickable-logo" onClick={onLogoClick}>
          Tech-nique
        </div>
      </header>

      {/* Completely empty content area for now */}
      <main className="upload-main">
        {/* intentionally empty – we’ll build this later */}
      </main>
    </div>
  );
}

export default UploadPage;
