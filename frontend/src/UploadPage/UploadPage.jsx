import React, { useEffect, useMemo, useRef, useState } from "react";
import { useNavigate } from "react-router-dom";
import "../App.css";
import "./UploadPage.css";

function UploadPage({ onLogoClick }) {
  const navigate = useNavigate();
  const apiBase = process.env.REACT_APP_API_BASE || "http://127.0.0.1:8000";
  const fileInputRef = useRef(null);

  const [exercises, setExercises] = useState([]);
  const [selectedExercise, setSelectedExercise] = useState("");
  const [selectedFile, setSelectedFile] = useState(null);
  const [dragActive, setDragActive] = useState(false);
  const [isUploading, setIsUploading] = useState(false);
  const [statusText, setStatusText] = useState("");

  useEffect(() => {
    const controller = new AbortController();

    async function loadExercises() {
      try {
        setStatusText("Loading exercises...");
        const response = await fetch(`${apiBase}/api/exercises`, {
          signal: controller.signal,
        });

        if (!response.ok) {
          throw new Error(`Failed to load exercises (${response.status})`);
        }

        const data = await response.json();
        const fetchedExercises = Array.isArray(data.exercises) ? data.exercises : [];
        setExercises(fetchedExercises);
        setSelectedExercise((current) => current || fetchedExercises[0] || "");
        setStatusText("");
      } catch (error) {
        if (error.name !== "AbortError") {
          setStatusText("Failed to load exercises from backend.");
        }
      }
    }

    loadExercises();
    return () => controller.abort();
  }, [apiBase]);

  const exerciseLabelMap = useMemo(
    () => ({
      bench_front: "Bench Front",
      bench_side: "Bench Side",
      squat_front: "Squat Front",
      squat_side: "Squat Side",
      pullup_back: "Pull-Up Back",
      bicep_curl_side: "Bicep Curl Side",
      tricep_extension_side: "Tricep Extension Side",
      shoulder_press_front: "Shoulder Press Front",
    }),
    []
  );

  function getExerciseLabel(exerciseKey) {
    return exerciseLabelMap[exerciseKey] || exerciseKey.replaceAll("_", " ");
  }

  function handleFilePick(file) {
    if (!file) {
      return;
    }

    if (!file.type.startsWith("video/")) {
      setStatusText("Please choose a video file.");
      return;
    }

    setSelectedFile(file);
    setStatusText("");
  }

  function onDrop(event) {
    event.preventDefault();
    setDragActive(false);

    const file = event.dataTransfer?.files?.[0];
    handleFilePick(file || null);
  }

  function onDragOver(event) {
    event.preventDefault();
    setDragActive(true);
  }

  function onDragLeave(event) {
    event.preventDefault();
    setDragActive(false);
  }

  async function onUpload() {
    if (!selectedExercise || !selectedFile) {
      setStatusText("Choose an exercise and a video first.");
      return;
    }

    const formData = new FormData();
    formData.append("exercise", selectedExercise);
    formData.append("video", selectedFile);

    setIsUploading(true);
    setStatusText("Uploading and analyzing video...");

    try {
      const response = await fetch(`${apiBase}/api/analyze`, {
        method: "POST",
        body: formData,
      });

      if (!response.ok) {
        const errorText = await response.text();
        throw new Error(errorText || `Upload failed (${response.status})`);
      }

      const data = await response.json();
      navigate("/result", {
        state: {
          exercise: data.exercise,
          analysis: data.analysis,
          feedbackVideoUrl: data.feedbackVideoUrl
            ? `${apiBase}${data.feedbackVideoUrl}`
            : "",
        },
      });
    } catch (error) {
      setStatusText(`Upload failed: ${error.message}`);
    } finally {
      setIsUploading(false);
    }
  }

  return (
    <div className="app upload-page">
      <header className="app-header">
        <div className="logo clickable-logo" onClick={onLogoClick}>
          Tech-nique
        </div>
      </header>

      <main className="upload-main">
        <section className="upload-card">
          <h1 className="upload-title">Upload Workout Video</h1>

          <div className="exercise-toggle-row">
            {exercises.map((exercise) => (
              <button
                key={exercise}
                type="button"
                className={`exercise-toggle-btn ${
                  selectedExercise === exercise ? "active" : ""
                }`}
                onClick={() => setSelectedExercise(exercise)}
              >
                {getExerciseLabel(exercise)}
              </button>
            ))}
          </div>

          <div
            className={`drop-zone ${dragActive ? "drag-active" : ""}`}
            onDrop={onDrop}
            onDragOver={onDragOver}
            onDragLeave={onDragLeave}
            onClick={() => fileInputRef.current?.click()}
            role="button"
            tabIndex={0}
            onKeyDown={(event) => {
              if (event.key === "Enter" || event.key === " ") {
                fileInputRef.current?.click();
              }
            }}
          >
            <p className="drop-zone-main">Drag a video here</p>
            <p className="drop-zone-sub">or click to choose a file</p>
            {selectedFile && (
              <p className="drop-zone-file">Selected: {selectedFile.name}</p>
            )}
          </div>

          <input
            ref={fileInputRef}
            className="hidden-file-input"
            type="file"
            accept="video/*"
            onChange={(event) => handleFilePick(event.target.files?.[0] || null)}
          />

          <button
            type="button"
            className="upload-submit-btn"
            onClick={onUpload}
            disabled={isUploading || !selectedExercise || !selectedFile}
          >
            {isUploading ? "Uploading..." : "Upload"}
          </button>

          {statusText && <p className="upload-status">{statusText}</p>}
        </section>
      </main>
    </div>
  );
}

export default UploadPage;
