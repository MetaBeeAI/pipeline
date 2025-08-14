// React webapp frontend
//
// m.mieskolainen@imperial.ac.uk, 2025

import React, { useEffect, useState, useRef, useCallback } from "react";
import Split from "react-split"; // Updated split library
import { Document, Page, pdfjs } from "react-pdf";
import "react-pdf/dist/esm/Page/AnnotationLayer.css";
import { AiOutlineStar, AiFillStar } from "react-icons/ai";

pdfjs.GlobalWorkerOptions.workerSrc = `//cdnjs.cloudflare.com/ajax/libs/pdf.js/${pdfjs.version}/pdf.worker.min.js`;

// Use the API_BASE_URL prefix for all API calls.
// For production, update this value as needed.
// const API_BASE_URL = "http://localhost:8000"
const API_BASE_URL = "/api";

// -------------------------
// LoginForm Component
// -------------------------
interface LoginFormProps {
  onLogin: (username: string, password: string) => Promise<void>;
}

function LoginForm({ onLogin }: LoginFormProps) {
  const [username, setUsername] = useState("");
  const [password, setPassword] = useState("");

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    await onLogin(username, password);
  };

  return (
    <div style={{ padding: "2rem", textAlign: "center" }}>
      <h2>Please Login</h2>
      <form onSubmit={handleSubmit}>
        <input
          type="text"
          placeholder="Username"
          value={username}
          onChange={(e) => setUsername(e.target.value)}
          required
          style={{ margin: "0.5rem", padding: "0.5rem", fontSize: "1rem" }}
        />
        <br />
        <input
          type="password"
          placeholder="Password"
          value={password}
          onChange={(e) => setPassword(e.target.value)}
          required
          style={{ margin: "0.5rem", padding: "0.5rem", fontSize: "1rem" }}
        />
        <br />
        <button type="submit" style={{ padding: "0.5rem 1rem", fontSize: "1rem" }}>
          Login
        </button>
      </form>
    </div>
  );
}

// -------------------------
// Interfaces & Types
// -------------------------
interface PaperData {
  id: string;
  progress?: number;
}

interface Chunk {
  chunk_id: string;
  grounding: {
    page: number;
    box: { l: number; t: number; r: number; b: number };
  }[];
}

interface QuestionObj {
  answer?: string;
  reason?: string;
  chunk_ids?: (string | number)[];
}

interface AnswersExtendedEntry {
  user_answer_positive: string;
  user_answer_negative: string;
  user_reason_positive: string;
  user_reason_negative: string;
  user_rating: number;
}

interface AnswersExtended {
  QUESTIONS: Record<string, AnswersExtendedEntry>;
}

interface AnswersData {
  QUESTIONS: Record<string, any>;
}

interface PdfBoxAnnotation {
  cid: string;
  box: { l: number; t: number; r: number; b: number };
  page: number;
}

enum AnnotationMode {
  ALL = "all",
  INDIVIDUAL = "individual",
}

type Theme = "dark" | "light";

interface Credentials {
  username: string;
  password: string;
}

// -------------------------
// StarRating Component
// -------------------------
function StarRating({
  maxStars = 10,
  value,
  onChange,
}: {
  maxStars?: number;
  value: number;
  onChange: (val: number) => void;
}) {
  const handleClick = (starIndex: number) => {
    onChange(starIndex === value ? 0 : starIndex);
  };

  return (
    <div style={{ display: "inline-flex", alignItems: "center", gap: "0.25rem" }}>
      {Array.from({ length: maxStars }).map((_, i) => {
        const starIndex = i + 1;
        return (
          <span
            key={starIndex}
            onClick={() => handleClick(starIndex)}
            style={{ cursor: "pointer", fontSize: "1.5rem", lineHeight: 1 }}
          >
            {starIndex <= value ? <AiFillStar /> : <AiOutlineStar />}
          </span>
        );
      })}
      <span style={{ marginLeft: "0.25rem" }}>{value}</span>
    </div>
  );
}

// Helper: check if an AnswersExtendedEntry is all empty (default)
function isDefaultEntry(entry: AnswersExtendedEntry) {
  return (
    !entry.user_answer_positive.trim() &&
    !entry.user_answer_negative.trim() &&
    !entry.user_reason_positive.trim() &&
    !entry.user_reason_negative.trim() &&
    Number(entry.user_rating) === 0
  );
}

// -------------------------
// MainApp Component (UI after login)
// -------------------------
interface MainAppProps {
  credentials: Credentials;
}

function MainApp({ credentials }: MainAppProps) {
  // UI states
  const [theme, setTheme] = useState("dark" as Theme);
  const [fontSize, setFontSize] = useState<number>(12);
  const [papers, setPapers] = useState<PaperData[]>([]);
  const [selectedPaper, setSelectedPaper] = useState<string>("");

  // PDF state
  const [pdfInstance, setPdfInstance] = useState<any>(null);
  const [pdfUrl, setPdfUrl] = useState<string>("");
  const [numPages, setNumPages] = useState<number>(0);
  const [currentPageIndex, setCurrentPageIndex] = useState<number>(0);
  const [zoom, setZoom] = useState<number>(100);
  const [baseScale, setBaseScale] = useState<number>(1);
  const pdfWrapperRef = useRef<HTMLDivElement>(null);

  // Data states
  const [chunkDict, setChunkDict] = useState<Record<string, Chunk>>({});
  const [answersData, setAnswersData] = useState<AnswersData>({ QUESTIONS: {} });
  const [answersExtended, setAnswersExtended] = useState<AnswersExtended>({ QUESTIONS: {} });
  const [questionMap, setQuestionMap] = useState<Record<string, QuestionObj>>({});
  const [selectedQuestionId, setSelectedQuestionId] = useState<string>("");

  const [modifiedLabel, setModifiedLabel] = useState<string>("Modified: Never");
  const [annotationMode, setAnnotationMode] = useState<AnnotationMode>(AnnotationMode.ALL);
  const [selectedChunkId, setSelectedChunkId] = useState<string>("");

  // User editing fields
  const [userAnswerPositive, setUserAnswerPositive] = useState("");
  const [userAnswerNegative, setUserAnswerNegative] = useState("");
  const [userReasonPositive, setUserReasonPositive] = useState("");
  const [userReasonNegative, setUserReasonNegative] = useState("");
  const [userRating, setUserRating] = useState(0);

  // AI read-only fields
  const [aiAnswer, setAiAnswer] = useState("");
  const [aiReason, setAiReason] = useState("");

  // For question → chunk mapping
  const [chunkIdsForQuestion, setChunkIdsForQuestion] = useState<string[]>([]);
  const [isLoadingQuestion, setIsLoadingQuestion] = useState<boolean>(false);

  // -------------------------
  // Effects for Theme & Font
  // -------------------------
  useEffect(() => {
    document.documentElement.setAttribute("data-theme", theme);
    document.body.style.color = theme === "dark" ? "#d4d4d4" : "#000000";
  }, [theme]);

  useEffect(() => {
    document.body.style.fontSize = `${fontSize}px`;
  }, [fontSize]);

  // Color scheme definitions.
  const colors = {
    dark: {
      listBg: "#000000",
      listColor: "#d4d4d4",
      buttonBg: "#3c3c3c",
      buttonColor: "#d4d4d4",
      centerBg: "#000000",
      rightBg: "#000000",
      rightColor: "#d4d4d4",
      border: "none",
    },
    light: {
      listBg: "#ffffff",
      listColor: "#333333",
      buttonBg: "#e7e7e7",
      buttonColor: "#333333",
      centerBg: "#f3f3f3",
      rightBg: "#ffffff",
      rightColor: "#333333",
      border: "none",
    },
  };

  // -------------------------
  // 1) Fetch papers.
  // -------------------------
  useEffect(() => {
    async function fetchPapers() {
      try {
        const res = await fetch(`${API_BASE_URL}/papers`);
        const data = await res.json();
        const loaded: PaperData[] =
          data.papers?.map((id: string) => ({ id, progress: 0 })) || [];
        setPapers(loaded);
      } catch (err) {
        console.error("Failed to fetch papers:", err);
      }
    }
    fetchPapers();
  }, []);

  // -------------------------
  // Scroll active items into view.
  // -------------------------
  useEffect(() => {
    if (selectedPaper) {
      const el = document.getElementById(`paper-${selectedPaper}`);
      if (el) el.scrollIntoView({ block: "nearest", behavior: "smooth" });
    }
  }, [selectedPaper]);

  useEffect(() => {
    if (selectedQuestionId) {
      const el = document.getElementById(`question-${selectedQuestionId}`);
      if (el) el.scrollIntoView({ block: "nearest", behavior: "smooth" });
    }
  }, [selectedQuestionId]);

  useEffect(() => {
    if (selectedChunkId) {
      const el = document.getElementById(`chunk-${selectedChunkId}`);
      if (el) el.scrollIntoView({ block: "nearest", behavior: "smooth" });
    }
  }, [selectedChunkId]);

  // -------------------------
  // Keyboard navigation for lists.
  // -------------------------
  const handlePaperKeyDown = (e: React.KeyboardEvent<HTMLDivElement>) => {
    if (papers.length === 0) return;
    const currentIndex = papers.findIndex((p) => p.id === selectedPaper);
    if (e.key === "ArrowDown") {
      const nextIndex = currentIndex === -1 ? 0 : Math.min(currentIndex + 1, papers.length - 1);
      handleSelectPaper(papers[nextIndex].id);
      e.preventDefault();
    } else if (e.key === "ArrowUp") {
      const prevIndex = currentIndex === -1 ? 0 : Math.max(currentIndex - 1, 0);
      handleSelectPaper(papers[prevIndex].id);
      e.preventDefault();
    }
  };

  const handleQuestionKeyDown = (e: React.KeyboardEvent<HTMLDivElement>) => {
    const questionKeysArr = Object.keys(questionMap);
    if (questionKeysArr.length === 0) return;
    const currentIndex = questionKeysArr.findIndex((q) => q === selectedQuestionId);
    if (e.key === "ArrowDown") {
      const nextIndex = currentIndex === -1 ? 0 : Math.min(currentIndex + 1, questionKeysArr.length - 1);
      handleSelectQuestion(questionKeysArr[nextIndex]);
      e.preventDefault();
    } else if (e.key === "ArrowUp") {
      const prevIndex = currentIndex === -1 ? 0 : Math.max(currentIndex - 1, 0);
      handleSelectQuestion(questionKeysArr[prevIndex]);
      e.preventDefault();
    }
  };

  const handleChunkKeyDown = (e: React.KeyboardEvent<HTMLDivElement>) => {
    if (chunkIdsForQuestion.length === 0) return;
    const currentIndex = chunkIdsForQuestion.findIndex((cid) => cid === selectedChunkId);
    if (e.key === "ArrowDown") {
      const nextIndex = currentIndex === -1 ? 0 : Math.min(currentIndex + 1, chunkIdsForQuestion.length - 1);
      const newChunkId = chunkIdsForQuestion[nextIndex];
      setSelectedChunkId(newChunkId);
      const chunk = chunkDict[newChunkId];
      if (chunk?.grounding?.length) setCurrentPageIndex(chunk.grounding[0].page);
      e.preventDefault();
    } else if (e.key === "ArrowUp") {
      const prevIndex = currentIndex === -1 ? 0 : Math.max(currentIndex - 1, 0);
      const newChunkId = chunkIdsForQuestion[prevIndex];
      setSelectedChunkId(newChunkId);
      const chunk = chunkDict[newChunkId];
      if (chunk?.grounding?.length) setCurrentPageIndex(chunk.grounding[0].page);
      e.preventDefault();
    }
  };

  // -------------------------
  // 2) Selecting a paper: load PDF, JSON, and answers.
  // -------------------------
  const handleSelectPaper = useCallback(async (paperId: string) => {
    if (!paperId) return;
    // Reset state for the new paper.
    setSelectedPaper(paperId);
    setSelectedQuestionId("");
    setQuestionMap({});
    setUserAnswerPositive("");
    setUserAnswerNegative("");
    setUserReasonPositive("");
    setUserReasonNegative("");
    setUserRating(0);
    setAiAnswer("");
    setAiReason("");
    setChunkIdsForQuestion([]);
    setChunkDict({});

    // Set PDF URL and reset page counters.
    const pdfURL = `${API_BASE_URL}/static/${paperId}/${paperId}_main.pdf`;
    setPdfUrl(pdfURL);
    setNumPages(0);
    setCurrentPageIndex(0);

    // Fetch JSON data (chunks/questions)
    try {
      const jsonRes = await fetch(`${API_BASE_URL}/json/${paperId}`);
      const jsonData = await jsonRes.json();
      const chunks = jsonData.data?.chunks || [];
      const cdict: { [key: string]: Chunk } = {};
      chunks.forEach((ch: Chunk) => {
        if (ch.chunk_id) {
          cdict[ch.chunk_id.toString().trim()] = ch;
        }
      });
      setChunkDict(cdict);
    } catch (err) {
      console.error("Detailed JSON fetch error:", err);
    }

    // Fetch official answers.
    try {
      const ansRes = await fetch(`${API_BASE_URL}/answers/${paperId}`);
      const ansData = await ansRes.json();
      setAnswersData(ansData);
      const questionsObj = ansData.QUESTIONS || {};
      const newQMap: { [key: string]: QuestionObj } = {};
      Object.keys(questionsObj).forEach((qKey) => {
        newQMap[qKey] = {
          answer: questionsObj[qKey].answer,
          reason: questionsObj[qKey].reason,
          chunk_ids: questionsObj[qKey].chunk_ids,
        };
      });
      setQuestionMap(newQMap);
    } catch (err) {
      console.error("Failed to fetch answers data:", err);
      setAnswersData({ QUESTIONS: {} });
    }

    // Fetch extended answers (requires authentication).
    try {
      const extRes = await fetch(
        `${API_BASE_URL}/answers_extended/${paperId}?username=${encodeURIComponent(
          credentials.username
        )}&password=${encodeURIComponent(credentials.password)}`
      );
      if (extRes.ok) {
        const extData = await extRes.json();
        setAnswersExtended(extData);
        if (extData.modified_timestamp) {
          setModifiedLabel(`Modified: ${extData.modified_timestamp}`);
        } else {
          setModifiedLabel("Modified: Never");
        }
      } else {
        setAnswersExtended({ QUESTIONS: {} });
        setModifiedLabel("Modified: Never");
      }
    } catch (err) {
      console.error("Failed to fetch extended answers:", err);
      setAnswersExtended({ QUESTIONS: {} });
      setModifiedLabel("Modified: Never");
    }

    // Reset annotation mode and selected chunk.
    setAnnotationMode(AnnotationMode.ALL);
    setSelectedChunkId("");
  }, [credentials]);

  // -------------------------
  // 3) PDF events.
  // -------------------------
  const onDocumentLoadSuccess = async (pdf: any) => {
    setPdfInstance(pdf);
    setNumPages(pdf.numPages);
    setCurrentPageIndex(0);
    const page = await pdf.getPage(1);
    const viewport = page.getViewport({ scale: 1 });
    const containerHeight = pdfWrapperRef.current?.clientHeight || window.innerHeight;
    const newBaseScale = containerHeight / viewport.height;
    setBaseScale(newBaseScale);
  };

  useEffect(() => {
    if (zoom === 100 && pdfInstance && pdfWrapperRef.current) {
      const resizeObserver = new ResizeObserver(() => {
        const containerHeight = pdfWrapperRef.current!.clientHeight;
        pdfInstance.getPage(1).then((page: any) => {
          const viewport = page.getViewport({ scale: 1 });
          const newBaseScale = containerHeight / viewport.height;
          setBaseScale(newBaseScale);
        });
      });
      resizeObserver.observe(pdfWrapperRef.current);
      return () => resizeObserver.disconnect();
    }
  }, [zoom, pdfInstance]);

  const goToPrevPage = () => {
    setCurrentPageIndex((p) => Math.max(p - 1, 0));
  };
  const goToNextPage = () => {
    setCurrentPageIndex((p) => Math.min(p + 1, numPages - 1));
  };

  // -------------------------
  // 4) Get bounding boxes for the current page.
  // -------------------------
  const getActiveAnnotations = (): PdfBoxAnnotation[] => {
    if (!selectedQuestionId) {
      const boxes: PdfBoxAnnotation[] = [];
      Object.values(chunkDict).forEach((chunk) => {
        chunk.grounding.forEach((g) => {
          if (g.page === currentPageIndex) {
            boxes.push({ cid: chunk.chunk_id, box: g.box, page: g.page });
          }
        });
      });
      return boxes;
    } else if (annotationMode === AnnotationMode.INDIVIDUAL && selectedChunkId) {
      const singleChunk = chunkDict[selectedChunkId];
      if (!singleChunk) return [];
      return singleChunk.grounding
        .filter((g) => g.page === currentPageIndex)
        .map((g) => ({ cid: singleChunk.chunk_id, box: g.box, page: g.page }));
    } else {
      const all: PdfBoxAnnotation[] = [];
      chunkIdsForQuestion.forEach((cid) => {
        const c = chunkDict[cid];
        if (!c) return;
        c.grounding.forEach((g) => {
          if (g.page === currentPageIndex) {
            all.push({ cid: c.chunk_id, box: g.box, page: g.page });
          }
        });
      });
      return all;
    }
  };

  // -------------------------
  // 5) Handle question selection.
  // -------------------------
  const questionKeys = Object.keys(questionMap);
  const handleSelectQuestion = (qId: string) => {
    if (qId === selectedQuestionId) return;
    setIsLoadingQuestion(true);
    setSelectedQuestionId(qId);
    const qObj = questionMap[qId] || {};
    setAiAnswer(qObj.answer || "");
    setAiReason(qObj.reason || "");
    setUserAnswerPositive("");
    setUserAnswerNegative("");
    setUserReasonPositive("");
    setUserReasonNegative("");
    setUserRating(0);
    const extEntry = answersExtended.QUESTIONS[qId];
    if (extEntry) {
      setUserAnswerPositive(extEntry.user_answer_positive || "");
      setUserAnswerNegative(extEntry.user_answer_negative || "");
      setUserReasonPositive(extEntry.user_reason_positive || "");
      setUserReasonNegative(extEntry.user_reason_negative || "");
      setUserRating(extEntry.user_rating || 0);
    }
    const cids = qObj.chunk_ids?.map((x) => String(x).trim()) || [];
    setChunkIdsForQuestion(cids);
    for (const cid of cids) {
      const chunk = chunkDict[cid];
      if (chunk?.grounding?.length) {
        setCurrentPageIndex(chunk.grounding[0].page);
        break;
      }
    }
    setAnnotationMode(AnnotationMode.ALL);
    setSelectedChunkId("");
    setIsLoadingQuestion(false);
  };

  // -------------------------
  // 6) Selecting a chunk (individual mode)
  // -------------------------
  const handleSelectChunk = (cid: string) => {
    if (annotationMode === AnnotationMode.ALL) return;
    setSelectedChunkId(cid);
    const chunk = chunkDict[cid];
    if (chunk?.grounding?.length) {
      setCurrentPageIndex(chunk.grounding[0].page);
    }
  };

  // -------------------------
  // 7) Auto-save user data.
  // -------------------------
  const handleAutoSave = useCallback(() => {
    if (!selectedQuestionId || isLoadingQuestion) return;
    const newData: AnswersExtendedEntry = {
      user_answer_positive: userAnswerPositive.trim(),
      user_answer_negative: userAnswerNegative.trim(),
      user_reason_positive: userReasonPositive.trim(),
      user_reason_negative: userReasonNegative.trim(),
      user_rating: userRating,
    };
    const defaultData: AnswersExtendedEntry = {
      user_answer_positive: "",
      user_answer_negative: "",
      user_reason_positive: "",
      user_reason_negative: "",
      user_rating: 0,
    };
    const oldData = answersExtended.QUESTIONS[selectedQuestionId] || defaultData;
    if (JSON.stringify(newData) === JSON.stringify(oldData)) return;
    const newAnswersExtended = JSON.parse(JSON.stringify(answersExtended));
    newAnswersExtended.QUESTIONS[selectedQuestionId] = newData;
    setAnswersExtended(newAnswersExtended);
    if (isDefaultEntry(newData)) {
      saveAnswersExtended(newAnswersExtended, true);
      return;
    }
    saveAnswersExtended(newAnswersExtended, false);
  }, [
    selectedQuestionId,
    userAnswerPositive,
    userAnswerNegative,
    userReasonPositive,
    userReasonNegative,
    userRating,
    isLoadingQuestion,
    answersExtended,
    selectedPaper,
  ]);

  async function saveAnswersExtended(data: AnswersExtended, isDefault: boolean) {
    try {
      const resp = await fetch(
        `${API_BASE_URL}/answers_extended/${selectedPaper}?username=${encodeURIComponent(
          credentials.username
        )}&password=${encodeURIComponent(credentials.password)}`,
        {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify(data),
        }
      );
      if (resp.ok) {
        const resJson = await resp.json();
        if (resJson.modified_timestamp) {
          setModifiedLabel(`Modified: ${resJson.modified_timestamp}`);
        } else {
          const modTime = new Date().toISOString().replace("T", " ").split(".")[0];
          setModifiedLabel(`Modified: ${modTime}`);
        }
      } else {
        console.error("Failed to save extended answers:", resp.statusText);
      }
    } catch (err) {
      console.error("Error saving extended answers:", err);
    }
    updatePaperProgress(selectedPaper);
  }

  // -------------------------
  // 8) Progress logic.
  // -------------------------
  function updatePaperProgress(paperId: string) {
    const questionKeys = Object.keys(answersData.QUESTIONS || {});
    const totalQuestions = questionKeys.length;
    if (totalQuestions === 0) {
      setPapers((prev) =>
        prev.map((p) => (p.id === paperId ? { ...p, progress: 0 } : p))
      );
      return;
    }
    const totalFields = totalQuestions * 5;
    let filled = 0;
    for (const qid of questionKeys) {
      const ext = answersExtended.QUESTIONS[qid] || {
        user_answer_positive: "",
        user_answer_negative: "",
        user_reason_positive: "",
        user_reason_negative: "",
        user_rating: 0,
      };
      if (ext.user_answer_positive.trim()) filled++;
      if (ext.user_answer_negative.trim()) filled++;
      if (ext.user_reason_positive.trim()) filled++;
      if (ext.user_reason_negative.trim()) filled++;
      if (Number(ext.user_rating) !== 0) filled++;
    }
    const pct = Math.floor((filled / totalFields) * 100);
    setPapers((prev) =>
      prev.map((p) => (p.id === paperId ? { ...p, progress: pct } : p))
    );
  }

  useEffect(() => {
    if (selectedPaper) {
      updatePaperProgress(selectedPaper);
    }
  }, [selectedPaper, answersExtended, answersData]);

  useEffect(() => {
    handleAutoSave();
  }, [
    userAnswerPositive,
    userAnswerNegative,
    userReasonPositive,
    userReasonNegative,
    userRating,
    handleAutoSave,
  ]);

  // -------------------------
  // Render the main UI.
  // -------------------------
  return (
    <div
      style={{
        margin: 0,
        padding: 0,
        height: "100vh",
        overflow: "hidden",
        display: "flex",
        fontFamily: "Futura, sans-serif",
      }}
    >
      {/* Outer Split: Left sidebar (20%) and Main Content (80%) */}
      <Split
        sizes={[20, 80]}
        minSize={[10, 10]}
        gutterSize={8}
        gutterStyle={() => ({
          cursor: "col-resize",
          backgroundColor: "#00008B",
          width: "0.5rem",
        })}
        direction="horizontal"
        style={{ display: "flex", width: "100%", height: "100vh" }}
      >
        {/* Left Sidebar */}
        <div
          style={{
            display: "flex",
            flexDirection: "column",
            height: "100%",
            backgroundColor: theme === "dark" ? colors.dark.listBg : colors.light.listBg,
            borderRight: theme === "dark" ? colors.dark.border : colors.light.border,
          }}
        >
          {/* Header */}
          <div
            style={{
              padding: "0.5rem",
              textAlign: "center",
              fontWeight: "bold",
              fontSize: "1.3rem",
              color: theme === "dark" ? "#f5eaa2" : "#93afdb",
            }}
          >
            MetaBeeAI
          </div>
          {/* Toolbar for Dark Mode and Font Size */}
          <div
            style={{
              display: "flex",
              gap: "0.5rem",
              padding: "0.5rem",
              justifyContent: "center",
            }}
          >
            <select
              value={theme}
              onChange={(e) => setTheme(e.target.value as Theme)}
              style={{
                padding: "0.25rem",
                backgroundColor: theme === "dark" ? colors.dark.buttonBg : colors.light.buttonBg,
                color: theme === "dark" ? colors.dark.buttonColor : colors.light.buttonColor,
                border: "none",
              }}
            >
              <option value="dark">Dark Mode</option>
              <option value="light">Light Mode</option>
            </select>
            <select
              value={fontSize}
              onChange={(e) => setFontSize(Number(e.target.value))}
              style={{
                padding: "0.25rem",
                backgroundColor: theme === "dark" ? colors.dark.buttonBg : colors.light.buttonBg,
                color: theme === "dark" ? colors.dark.buttonColor : colors.light.buttonColor,
                border: "none",
              }}
            >
              {[10, 12, 14, 16, 18, 20].map((size) => (
                <option key={size} value={size}>
                  Font {size}
                </option>
              ))}
            </select>
          </div>
          {/* Paper Navigation */}
          <div style={{ padding: "0.5rem" }}>
            <button
              style={{
                width: "100%",
                marginBottom: "0.5rem",
                backgroundColor: theme === "dark" ? colors.dark.buttonBg : colors.light.buttonBg,
                color: theme === "dark" ? colors.dark.buttonColor : colors.light.buttonColor,
                border: "none",
                padding: "0.5rem",
              }}
              onClick={() => {
                if (!selectedPaper) return;
                const idx = papers.findIndex((p) => p.id === selectedPaper);
                if (idx > 0) {
                  handleSelectPaper(papers[idx - 1].id);
                }
              }}
            >
              Prev Paper
            </button>
            <button
              style={{
                width: "100%",
                backgroundColor: theme === "dark" ? colors.dark.buttonBg : colors.light.buttonBg,
                color: theme === "dark" ? colors.dark.buttonColor : colors.light.buttonColor,
                border: "none",
                padding: "0.5rem",
              }}
              onClick={() => {
                if (!selectedPaper) return;
                const idx = papers.findIndex((p) => p.id === selectedPaper);
                if (idx >= 0 && idx < papers.length - 1) {
                  handleSelectPaper(papers[idx + 1].id);
                }
              }}
            >
              Next Paper
            </button>
          </div>
          {/* Paper List */}
          <div
            tabIndex={0}
            onKeyDown={handlePaperKeyDown}
            style={{
              flex: 1,
              overflowY: "auto",
              padding: "0.5rem",
              borderTop: theme === "dark" ? colors.dark.border : colors.light.border,
            }}
          >
            {papers.map((paper) => {
              const label = paper.progress ? `${paper.id} (${paper.progress}%)` : paper.id;
              const isSelected = paper.id === selectedPaper;
              return (
                <div
                  id={`paper-${paper.id}`}
                  key={paper.id}
                  onClick={() => handleSelectPaper(paper.id)}
                  style={{
                    padding: "0.25rem 0.5rem",
                    cursor: "pointer",
                    backgroundColor: isSelected ? "#1E90FF" : "transparent",
                    color: isSelected
                      ? "#fff"
                      : theme === "dark"
                      ? colors.dark.listColor
                      : colors.light.listColor,
                  }}
                >
                  {label}
                </div>
              );
            })}
          </div>
          {/* PDF Controls & Toolbars */}
          <div style={{ borderTop: theme === "dark" ? colors.dark.border : colors.light.border, padding: "0.5rem" }}>
            <div style={{ display: "flex", alignItems: "center", justifyContent: "space-between", marginBottom: "0.5rem" }}>
              <button
                onClick={goToPrevPage}
                style={{
                  backgroundColor: theme === "dark" ? colors.dark.buttonBg : colors.light.buttonBg,
                  color: theme === "dark" ? colors.dark.buttonColor : colors.light.buttonColor,
                  border: "none",
                  padding: "0.25rem 0.5rem",
                }}
              >
                Prev
              </button>
              <div>
                Page {currentPageIndex + 1} / {numPages}
              </div>
              <button
                onClick={goToNextPage}
                style={{
                  backgroundColor: theme === "dark" ? colors.dark.buttonBg : colors.light.buttonBg,
                  color: theme === "dark" ? colors.dark.buttonColor : colors.light.buttonColor,
                  border: "none",
                  padding: "0.25rem 0.5rem",
                }}
              >
                Next
              </button>
            </div>
            <div style={{ display: "flex", alignItems: "center", marginBottom: "0.5rem" }}>
              <span>{zoom}%</span>
              <input
                type="range"
                min={10}
                max={400}
                step={10}
                value={zoom}
                onChange={(e) => setZoom(Number(e.target.value))}
                style={{ marginLeft: "0.5rem", flex: 1 }}
              />
            </div>
            {/* Display active username above Modified label */}
            <div
              style={{
                marginBottom: "0.5rem",
                color: theme === "dark" ? "#f5eaa2" : "#93afdb",
              }}
            >
              User: {credentials.username}
            </div>
            <div
              style={{
                marginBottom: "0.5rem",
                color: theme === "dark" ? "#f5eaa2" : "#93afdb",
              }}
            >
              {modifiedLabel}
            </div>
          </div>
        </div>

        {/* Inner Split: Center (PDF view) and Right Sidebar (Questions/Chunks) */}
        <Split
          sizes={[70, 30]}
          minSize={[20, 20]}
          gutterSize={8}
          gutterStyle={() => ({
            cursor: "col-resize",
            backgroundColor: "#00008B",
            width: "0.5rem",
          })}
          direction="horizontal"
          style={{ display: "flex", width: "100%", height: "100%" }}
        >
          {/* Center: PDF view */}
          <div style={{ position: "relative", height: "100vh" }}>
            <div
              ref={pdfWrapperRef}
              style={{
                position: "relative",
                height: "100vh",
                overflow: "auto",
                backgroundColor: theme === "dark" ? colors.dark.centerBg : colors.light.centerBg,
                display: "flex",
                justifyContent: "center",
                alignItems: "flex-start",
                padding: "0.5rem",
              }}
            >
              {pdfUrl ? (
                <Document file={pdfUrl} onLoadSuccess={onDocumentLoadSuccess} loading={""} error={""}>
                  <div style={{ position: "relative", display: "inline-block", border: "none", outline: "none" }}>
                    <Page
                      pageNumber={currentPageIndex + 1}
                      scale={baseScale * (zoom / 100)}
                      renderTextLayer={false}
                      renderAnnotationLayer={false}
                    />
                    {getActiveAnnotations().map((ann, i) => {
                      const styleBox: React.CSSProperties = {
                        position: "absolute",
                        border: "0.125rem solid red",
                        backgroundColor: "rgba(255,0,0,0.1)",
                        pointerEvents: "auto",
                        left: `calc(${ann.box.l * 100}% - 2px)`,
                        top: `calc(${ann.box.t * 100}% - 2px)`,
                        width: `calc(${(ann.box.r - ann.box.l) * 100}% + 4px)`,
                        height: `calc(${(ann.box.b - ann.box.t) * 100}% + 4px)`,
                      };
                      return <div key={i} style={styleBox} title={`Chunk: ${ann.cid}`} />;
                    })}
                  </div>
                </Document>
              ) : (
                <div style={{ height: "100%" }} />
              )}
            </div>
          </div>
          
          {/* Right Sidebar: Questions, Question Details, and Chunk List */}
          <div
            style={{
              minHeight: "100vh",
              backgroundColor: theme === "dark" ? colors.dark.listBg : colors.light.listBg,
              color: theme === "dark" ? colors.dark.listColor : colors.light.listColor,
              borderLeft: theme === "dark" ? colors.dark.border : colors.light.border,
              display: "flex",
              flexDirection: "column",
              overflowY: "auto",
            }}
          >
            {/* Questions List */}
            <div style={{ padding: "0.5rem", borderBottom: theme === "dark" ? colors.dark.border : colors.light.border }}>
              <div
                style={{
                  fontWeight: "bold",
                  textAlign: "center",
                  marginBottom: "0.5rem",
                  fontSize: "1.3rem",
                  color: theme === "dark" ? "#f5eaa2" : "#93afdb",
                }}
              >
                Questions
              </div>
              <div
                onKeyDown={handleQuestionKeyDown}
                tabIndex={0}
                style={{
                  maxHeight: "15vh",
                  overflowY: "auto",
                  border: theme === "dark" ? "0.125rem solid #666" : "0.125rem solid #aaa",
                  padding: "0.25rem",
                }}
              >
                {questionKeys.length === 0 ? (
                  <div style={{ textAlign: "center", opacity: 0.7 }}>(No questions)</div>
                ) : (
                  questionKeys.map((qId) => {
                    const isSelected = qId === selectedQuestionId;
                    return (
                      <div
                        id={`question-${qId}`}
                        key={qId}
                        onClick={() => handleSelectQuestion(qId)}
                        style={{
                          padding: "0.25rem",
                          cursor: "pointer",
                          backgroundColor: isSelected ? "#1E90FF" : "transparent",
                          color: isSelected ? "#fff" : undefined,
                        }}
                      >
                        {qId}
                      </div>
                    );
                  })
                )}
              </div>
            </div>
            {/* Question Details */}
            {selectedQuestionId && (
              <div style={{ flex: 1, display: "flex", flexDirection: "column", backgroundColor: theme === "dark" ? colors.dark.listBg : colors.light.listBg }}>
                <div style={{ padding: "0.5rem", flex: 1, overflowY: "auto" }}>
                  <div>
                    <label style={{ fontWeight: "bold", fontSize: "1.0rem" }}>AI Answer</label>
                    <textarea
                      style={{
                        width: "100%",
                        height: "8rem",
                        backgroundColor: theme === "dark" ? "#000000" : undefined,
                        color: theme === "dark" ? "#d4d4d4" : undefined,
                        marginBottom: "0.4rem",
                      }}
                      readOnly
                      value={aiAnswer}
                    />
                  </div>
                  <div>
                    <label style={{ fontWeight: "bold", fontSize: "1.0rem" }}>AI Reason</label>
                    <textarea
                      style={{
                        width: "100%",
                        height: "9rem",
                        backgroundColor: theme === "dark" ? "#000000" : undefined,
                        color: theme === "dark" ? "#d4d4d4" : undefined,
                        marginBottom: "0.4rem",
                      }}
                      readOnly
                      value={aiReason}
                    />
                  </div>
                  
                  <div style={{ marginTop: "0.25rem" }}>
                    <StarRating value={userRating} onChange={setUserRating} />
                  </div>

                  <div style={{ display: "flex", gap: "0.5rem", marginTop: "0.5rem" }}>
                    <div style={{ flex: 1 }}>
                      <label style={{ color: "green", fontSize: "0.9rem" }}>Answer (✓)</label>
                      <textarea
                        style={{
                          width: "100%",
                          height: "5rem",
                          backgroundColor: theme === "dark" ? "#000000" : undefined,
                          color: theme === "dark" ? "#d4d4d4" : undefined,
                        }}
                        value={userAnswerPositive}
                        onChange={(e) => setUserAnswerPositive(e.target.value)}
                      />
                    </div>
                    <div style={{ flex: 1 }}>
                      <label style={{ color: "green", fontSize: "0.9rem" }}>Reason (✓)</label>
                      <textarea
                        style={{
                          width: "100%",
                          height: "5rem",
                          backgroundColor: theme === "dark" ? "#000000" : undefined,
                          color: theme === "dark" ? "#d4d4d4" : undefined,
                        }}
                        value={userReasonPositive}
                        onChange={(e) => setUserReasonPositive(e.target.value)}
                      />
                    </div>
                  </div>
                  <div style={{ display: "flex", gap: "0.5rem", marginTop: "0.5rem" }}>
                    <div style={{ flex: 1 }}>
                      <label style={{ color: "red", fontSize: "0.9rem" }}>Answer (✗)</label>
                      <textarea
                        style={{
                          width: "100%",
                          height: "5rem",
                          backgroundColor: theme === "dark" ? "#000000" : undefined,
                          color: theme === "dark" ? "#d4d4d4" : undefined,
                        }}
                        value={userAnswerNegative}
                        onChange={(e) => setUserAnswerNegative(e.target.value)}
                      />
                    </div>
                    <div style={{ flex: 1 }}>
                      <label style={{ color: "red", fontSize: "0.9rem" }}>Reason (✗)</label>
                      <textarea
                        style={{
                          width: "100%",
                          height: "5rem",
                          backgroundColor: theme === "dark" ? "#000000" : undefined,
                          color: theme === "dark" ? "#d4d4d4" : undefined,
                        }}
                        value={userReasonNegative}
                        onChange={(e) => setUserReasonNegative(e.target.value)}
                      />
                    </div>
                  </div>
                </div>
              </div>
            )}
            {/* Chunk IDs List */}
            <div style={{ padding: "0.25rem", borderBottom: theme === "dark" ? colors.dark.border : colors.light.border }}>
              <div
                onKeyDown={handleChunkKeyDown}
                tabIndex={0}
                style={{
                  maxHeight: "10vh",
                  overflowY: "auto",
                  border: theme === "dark" ? "0.125rem solid #666" : "0.125rem solid #aaa",
                  padding: "0.125rem",
                }}
              >
                {selectedQuestionId
                  ? chunkIdsForQuestion.length > 0
                    ? chunkIdsForQuestion.map((cid) => {
                        const isSelected = cid === selectedChunkId;
                        return (
                          <div
                            id={`chunk-${cid}`}
                            key={cid}
                            style={{
                              padding: "0.25rem",
                              cursor:
                                annotationMode === AnnotationMode.INDIVIDUAL
                                  ? "pointer"
                                  : "default",
                              backgroundColor:
                                isSelected && annotationMode === AnnotationMode.INDIVIDUAL
                                  ? "#1E90FF"
                                  : "transparent",
                              color:
                                isSelected && annotationMode === AnnotationMode.INDIVIDUAL
                                  ? "#fff"
                                  : undefined,
                              opacity: annotationMode === AnnotationMode.ALL ? 0.5 : 1,
                            }}
                            onClick={() =>
                              annotationMode === AnnotationMode.INDIVIDUAL ? handleSelectChunk(cid) : null
                            }
                          >
                            {cid}
                          </div>
                        );
                      })
                    : <div style={{ textAlign: "center", opacity: 0.6 }}>(No chunks)</div>
                  : <div style={{ textAlign: "center", opacity: 0.6 }}>(No question selected)</div>}
              </div>
            </div>
            {/* Mode Buttons */}
            <div
              style={{
                padding: "0.5rem",
                borderTop: theme === "dark" ? colors.dark.border : colors.light.border,
                position: "sticky",
                bottom: 0,
                backgroundColor: theme === "dark" ? colors.dark.listBg : colors.light.listBg,
                zIndex: 10,
              }}
            >
              <div style={{ display: "flex", justifyContent: "center", gap: "0.5rem" }}>
                <button
                  onClick={() => setAnnotationMode(AnnotationMode.INDIVIDUAL)}
                  style={{
                    backgroundColor:
                      annotationMode === AnnotationMode.INDIVIDUAL ? "#1E90FF" : "transparent",
                    color:
                      annotationMode === AnnotationMode.INDIVIDUAL ? "#fff" : theme === "dark" ? colors.dark.buttonColor : colors.light.buttonColor,
                    border: "none",
                    padding: "0.25rem 0.5rem",
                    cursor: "pointer",
                    fontSize: "inherit",
                  }}
                >
                  Individual
                </button>
                <button
                  onClick={() => setAnnotationMode(AnnotationMode.ALL)}
                  style={{
                    backgroundColor:
                      annotationMode === AnnotationMode.ALL ? "#1E90FF" : "transparent",
                    color:
                      annotationMode === AnnotationMode.ALL ? "#fff" : theme === "dark" ? colors.dark.buttonColor : colors.light.buttonColor,
                    border: "none",
                    padding: "0.25rem 0.5rem",
                    cursor: "pointer",
                    fontSize: "inherit",
                  }}
                >
                  All
                </button>
              </div>
            </div>
          </div>
        </Split>
      </Split>
    </div>
  );
}

// -------------------------
// Top-level HomePageWrapper Component
// -------------------------
export default function HomePageWrapper() {
  const [credentials, setCredentials] = useState<Credentials | null>(null);
  const [loginError, setLoginError] = useState<string>("");

  const handleLogin = async (username: string, password: string) => {
    try {
      const res = await fetch(
        `${API_BASE_URL}/login?username=${encodeURIComponent(username)}&password=${encodeURIComponent(password)}`
      );
      if (!res.ok) {
        setLoginError("Invalid username or password");
      } else {
        setCredentials({ username, password });
        setLoginError("");
      }
    } catch (err) {
      setLoginError("An error occurred during login");
    }
  };

  return (
    <>
      {credentials ? (
        <MainApp credentials={credentials} />
      ) : (
        <div>
          <LoginForm onLogin={handleLogin} />
          {loginError && (
            <div style={{ color: "red", textAlign: "center", marginTop: "1rem" }}>
              {loginError}
            </div>
          )}
        </div>
      )}
    </>
  );
}
