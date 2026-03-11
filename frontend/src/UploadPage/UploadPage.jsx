import React, { useEffect, useMemo, useRef, useState } from "react";
import { useNavigate } from "react-router-dom";
import "../App.css";
import "./UploadPage.css";

const EXERCISE_METADATA = {
  bench_front: {
    muscleGroup: "Chest",
    exerciseName: "Bench Press",
    view: "Front",
  },
  bench_side: {
    muscleGroup: "Chest",
    exerciseName: "Bench Press",
    view: "Side",
  },
  squat_front: {
    muscleGroup: "Legs",
    exerciseName: "Squat",
    view: "Front",
  },
  squat_side: {
    muscleGroup: "Legs",
    exerciseName: "Squat",
    view: "Side",
  },
  pullup_back: {
    muscleGroup: "Back",
    exerciseName: "Pull-Up",
    view: "Back",
  },
  bicep_curl_side: {
    muscleGroup: "Arms",
    exerciseName: "Bicep Curl",
    view: "Side",
  },
  tricep_extension_side: {
    muscleGroup: "Arms",
    exerciseName: "Tricep Extension",
    view: "Side",
  },
  shoulder_press_front: {
    muscleGroup: "Shoulders",
    exerciseName: "Shoulder Press",
    view: "Front",
  },
};

function titleCaseExerciseKey(exerciseKey) {
  return exerciseKey
    .split("_")
    .map((part) => part.charAt(0).toUpperCase() + part.slice(1))
    .join(" ");
}

function UploadPage({ onLogoClick }) {
  const navigate = useNavigate();
  const apiBase = process.env.REACT_APP_API_BASE || "http://127.0.0.1:8000";
  const fileInputRef = useRef(null);

  const [exercises, setExercises] = useState([]);
  const [selectedMuscleGroup, setSelectedMuscleGroup] = useState("");
  const [selectedExerciseName, setSelectedExerciseName] = useState("");
  const [selectedView, setSelectedView] = useState("");
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

  const exerciseCatalog = useMemo(
    () =>
      exercises.map((exerciseKey) => ({
        key: exerciseKey,
        ...(EXERCISE_METADATA[exerciseKey] || {
          muscleGroup: "Other",
          exerciseName: titleCaseExerciseKey(exerciseKey),
          view: "Default",
        }),
      })),
    [exercises]
  );

  const muscleGroups = useMemo(
    () => [...new Set(exerciseCatalog.map((entry) => entry.muscleGroup))],
    [exerciseCatalog]
  );

  const exerciseNames = useMemo(() => {
    return [
      ...new Set(
        exerciseCatalog
          .filter((entry) => entry.muscleGroup === selectedMuscleGroup)
          .map((entry) => entry.exerciseName)
      ),
    ];
  }, [exerciseCatalog, selectedMuscleGroup]);

  const viewOptions = useMemo(() => {
    return exerciseCatalog
      .filter(
        (entry) =>
          entry.muscleGroup === selectedMuscleGroup &&
          entry.exerciseName === selectedExerciseName
      )
      .map((entry) => ({
        key: entry.key,
        label: entry.view,
      }));
  }, [exerciseCatalog, selectedExerciseName, selectedMuscleGroup]);

  const selectedExercise = useMemo(() => {
    return (
      viewOptions.find((option) => option.label === selectedView)?.key || ""
    );
  }, [selectedView, viewOptions]);

  useEffect(() => {
    if (!muscleGroups.length) {
      setSelectedMuscleGroup("");
      return;
    }

    setSelectedMuscleGroup((current) =>
      muscleGroups.includes(current) ? current : muscleGroups[0]
    );
  }, [muscleGroups]);

  useEffect(() => {
    if (!exerciseNames.length) {
      setSelectedExerciseName("");
      return;
    }

    setSelectedExerciseName((current) =>
      exerciseNames.includes(current) ? current : exerciseNames[0]
    );
  }, [exerciseNames]);

  useEffect(() => {
    if (!viewOptions.length) {
      setSelectedView("");
      return;
    }

    const validViews = viewOptions.map((option) => option.label);
    setSelectedView((current) =>
      validViews.includes(current) ? current : viewOptions[0].label
    );
  }, [viewOptions]);

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

          <div className="selection-stack">
            <div className="selection-group">
              <p className="selection-label">1. Muscle group</p>
              <div className="exercise-toggle-row">
                {muscleGroups.map((group) => (
                  <button
                    key={group}
                    type="button"
                    className={`exercise-toggle-btn ${
                      selectedMuscleGroup === group ? "active" : ""
                    }`}
                    onClick={() => setSelectedMuscleGroup(group)}
                  >
                    {group}
                  </button>
                ))}
              </div>
            </div>

            <div className="selection-group">
              <p className="selection-label">2. Exercise</p>
              <div className="exercise-toggle-row">
                {exerciseNames.map((exerciseName) => (
                  <button
                    key={exerciseName}
                    type="button"
                    className={`exercise-toggle-btn ${
                      selectedExerciseName === exerciseName ? "active" : ""
                    }`}
                    onClick={() => setSelectedExerciseName(exerciseName)}
                  >
                    {exerciseName}
                  </button>
                ))}
              </div>
            </div>

            <div className="selection-group">
              <p className="selection-label">3. Camera view</p>
              <div className="exercise-toggle-row">
                {viewOptions.map((viewOption) => (
                  <button
                    key={viewOption.key}
                    type="button"
                    className={`exercise-toggle-btn ${
                      selectedView === viewOption.label ? "active" : ""
                    }`}
                    onClick={() => setSelectedView(viewOption.label)}
                  >
                    {viewOption.label}
                  </button>
                ))}
              </div>
            </div>
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
