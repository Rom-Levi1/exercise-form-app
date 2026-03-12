import React from "react";
import { useLocation } from "react-router-dom";
import "../App.css";
import "../UploadPage/UploadPage.css";
import "./ResultPage.css";

const EXERCISE_LABELS = {
  squat_side: "Squat / Side View",
  squat_front: "Squat / Front View",
  bench_side: "Bench Press / Side View",
  bench_front: "Bench Press / Front View",
  bicep_curl_side: "Bicep Curl / Side View",
  tricep_extension_side: "Tricep Extension / Side View",
  shoulder_press_front: "Shoulder Press / Front View",
  pullup_back: "Pull-Up / Back View",
};

const RATING_LABELS = {
  strong: "Strong",
  good: "Good",
  needs_work: "Focus",
  poor: "Needs Work",
  unknown: "Unrated",
};

const SEVERITY_LABELS = {
  low: "Minor",
  medium: "Moderate",
  high: "Important",
};

function humanizeExercise(exerciseKey) {
  return (
    EXERCISE_LABELS[exerciseKey] ||
    exerciseKey
      .split("_")
      .map((part) => part.charAt(0).toUpperCase() + part.slice(1))
      .join(" ")
  );
}

function ResultPage({ onLogoClick, onBackToUpload }) {
  const location = useLocation();
  const payload = location.state || {};

  const exercise = payload.exercise || "";
  const analysis = payload.analysis || {};
  const textFeedback = payload.textFeedback || {};
  const feedbackVideoUrl = payload.feedbackVideoUrl || "";
  const overall = textFeedback.overall || {};
  const highlights = Array.isArray(textFeedback.highlights)
    ? textFeedback.highlights
    : [];
  const repBreakdown = Array.isArray(textFeedback.repBreakdown)
    ? textFeedback.repBreakdown
    : [];
  const warnings = Array.isArray(textFeedback.warnings) ? textFeedback.warnings : [];

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
                  <strong>Exercise:</strong> {humanizeExercise(exercise)}
                </p>
                <p>
                  <strong>Analysis:</strong> {feedbackVideoUrl ? "Complete" : "Unavailable"}
                </p>
                <p>
                  <strong>Rep Count:</strong>{" "}
                  {analysis.repCount !== undefined ? analysis.repCount : "-"}
                </p>
              </div>

              <div className="result-content-grid">
                <div className="video-wrap">
                  <video
                    className="result-video"
                    src={feedbackVideoUrl}
                    controls
                    autoPlay
                  />
                </div>

                <aside className="feedback-panel">
                  <div className="feedback-section">
                    <p className="feedback-eyebrow">Overall</p>
                    <h2 className="feedback-title">{overall.title || "Feedback summary"}</h2>
                    <p className="feedback-summary">
                      {overall.summary || "No text feedback is available for this exercise yet."}
                    </p>
                  </div>

                  <div className="feedback-section">
                    <p className="feedback-eyebrow">Main feedback</p>
                    <div className="feedback-highlight-list">
                      {highlights.map((item, index) => (
                        <article key={`${item.title}-${index}`} className="feedback-highlight-card">
                          <div className="feedback-highlight-top">
                            <h3>{item.title}</h3>
                            {item.severity ? (
                              <span className={`feedback-severity severity-${item.severity}`}>
                                {SEVERITY_LABELS[item.severity] || item.severity}
                              </span>
                            ) : null}
                          </div>
                          <p>{item.details}</p>
                          <p className="feedback-cue">
                            <strong>Coaching tip:</strong> {item.cue}
                          </p>
                          {Array.isArray(item.reps) && item.reps.length > 0 && (
                            <p className="feedback-reps">
                              Affected reps: {item.reps.join(", ")}
                            </p>
                          )}
                        </article>
                      ))}
                    </div>
                  </div>

                  {repBreakdown.length > 0 && (
                    <div className="feedback-section">
                      <p className="feedback-eyebrow">Rep breakdown</p>
                      <div className="rep-breakdown-list">
                        {repBreakdown.map((item) => (
                          <article key={item.rep} className="rep-breakdown-item">
                            <div className="rep-breakdown-top">
                              <strong>Rep {item.rep}</strong>
                              <span className={`feedback-severity severity-${item.rating || "unknown"}`}>
                                {RATING_LABELS[item.rating] || item.rating || "Unrated"}
                              </span>
                            </div>
                            <p>{item.label}</p>
                            <p className="rep-breakdown-detail">{item.details}</p>
                          </article>
                        ))}
                      </div>
                    </div>
                  )}

                  {warnings.length > 0 && (
                    <div className="feedback-section">
                      <p className="feedback-eyebrow">Warnings</p>
                      <div className="feedback-warning-list">
                        {warnings.map((warning) => (
                          <p key={warning} className="feedback-warning-item">
                            {warning}
                          </p>
                        ))}
                      </div>
                    </div>
                  )}
                </aside>
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
