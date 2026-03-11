import React from "react";
import { useLocation } from "react-router-dom";
import "../App.css";
import "../UploadPage/UploadPage.css";
import "./ResultPage.css";

function ResultPage({ onLogoClick, onBackToUpload }) {
  const location = useLocation();
  const payload = location.state || {};

  const exercise = payload.exercise || "";
  const analysis = payload.analysis || {};
  const feedbackVideoUrl = payload.feedbackVideoUrl || "";

  return (
    <div className="app upload-page">
      <header className="app-header">
        <div className="logo clickable-logo" onClick={onLogoClick}>
          Tech-nique
        </div>
      </header>

      <main className="upload-main">
        <section className="upload-card result-card">
          <h1 className="upload-title">Analysis Result</h1>

          {!feedbackVideoUrl ? (
            <div className="result-empty">
              <p>No analysis result was found for this page.</p>
            </div>
          ) : (
            <>
              <div className="result-meta">
                <p>
                  <strong>Exercise:</strong> {exercise}
                </p>
                <p>
                  <strong>Status:</strong> {analysis.status || "-"}
                </p>
                <p>
                  <strong>Rep Count:</strong>{" "}
                  {analysis.repCount !== undefined ? analysis.repCount : "-"}
                </p>
              </div>

              <div className="video-wrap">
                <video
                  className="result-video"
                  src={feedbackVideoUrl}
                  controls
                  autoPlay
                />
              </div>
            </>
          )}

          <button
            type="button"
            className="upload-submit-btn back-btn"
            onClick={onBackToUpload}
          >
            Back To Upload
          </button>
        </section>
      </main>
    </div>
  );
}

export default ResultPage;
