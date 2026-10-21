// Login / Signup page: HeartRisk AI logo, email + password, and a public
// "Create account" form. After a successful signup the user is logged in
// straight away (the signup endpoint returns a token).
import { useState } from "react";
import { useNavigate } from "react-router-dom";
import { apiFetch, saveLogin } from "../api";
import { HeartLogo } from "../components/Icon";

export default function Login() {
  const [mode, setMode] = useState("login");   // "login" | "signup"
  const [email, setEmail] = useState("");
  const [password, setPassword] = useState("");
  const [name, setName] = useState("");         // signup only
  const [confirm, setConfirm] = useState("");   // signup only
  const [error, setError] = useState("");
  const [success, setSuccess] = useState("");
  const [busy, setBusy] = useState(false);
  const navigate = useNavigate();

  function switchMode(next) {
    setMode(next);
    setError("");
    setSuccess("");
  }

  async function submitLogin(e) {
    e.preventDefault();
    setError("");
    setBusy(true);
    try {
      const data = await apiFetch("/api/auth/login", {
        method: "POST",
        body: JSON.stringify({ email, password }),
      });
      saveLogin(data);              // store token + role + name
      navigate("/");                // go to the dashboard
    } catch (err) {
      setError(err.message);        // e.g. "Wrong email or password"
    } finally {
      setBusy(false);
    }
  }

  async function submitSignup(e) {
    e.preventDefault();
    setError("");
    if (!name.trim()) { setError("Please enter your full name."); return; }
    if (password !== confirm) { setError("The two passwords do not match."); return; }
    if (password.length < 8) { setError("Password must be at least 8 characters."); return; }
    setBusy(true);
    try {
      const data = await apiFetch("/api/auth/signup", {
        method: "POST",
        body: JSON.stringify({ full_name: name.trim(), email, password }),
      });
      saveLogin(data);              // log the new doctor in right away
      navigate("/");
    } catch (err) {
      setError(err.message);        // e.g. "This email is already registered"
    } finally {
      setBusy(false);
    }
  }

  return (
    <div className="login-page">
      <div className="login-card">
        <div className="login-logo"><HeartLogo size={34} /></div>
        <h1>HeartRisk AI</h1>
        <div className="tagline">
          {mode === "login" ? "Please login to continue" : "Create your doctor account"}
        </div>

        {error && <div className="error-banner">{error}</div>}
        {success && <div className="info-banner">{success}</div>}

        {mode === "login" ? (
          <form onSubmit={submitLogin}>
            <label className="field">
              <span>Email</span>
              <input
                type="email"
                placeholder="doctor@hospital.com"
                value={email}
                onChange={(e) => setEmail(e.target.value)}
                required
              />
            </label>
            <label className="field" style={{ marginTop: 12 }}>
              <span>Password</span>
              <input
                type="password"
                placeholder="••••••••"
                value={password}
                onChange={(e) => setPassword(e.target.value)}
                required
              />
            </label>
            <button className="btn" disabled={busy}>
              {busy ? "Logging in..." : "Login"}
            </button>
            <div className="login-help">
              Don't have an account?{" "}
              <a href="#signup" onClick={(e) => { e.preventDefault(); switchMode("signup"); }}>
                Sign up
              </a>
            </div>
          </form>
        ) : (
          <form onSubmit={submitSignup}>
            <label className="field">
              <span>Full name</span>
              <input
                type="text"
                placeholder="Dr Jane Doe"
                value={name}
                onChange={(e) => setName(e.target.value)}
                required
              />
            </label>
            <label className="field" style={{ marginTop: 12 }}>
              <span>Email</span>
              <input
                type="email"
                placeholder="jane@hospital.com"
                value={email}
                onChange={(e) => setEmail(e.target.value)}
                required
              />
            </label>
            <label className="field" style={{ marginTop: 12 }}>
              <span>Password (min 8 characters)</span>
              <input
                type="password"
                placeholder="••••••••"
                value={password}
                onChange={(e) => setPassword(e.target.value)}
                required
              />
            </label>
            <label className="field" style={{ marginTop: 12 }}>
              <span>Confirm password</span>
              <input
                type="password"
                placeholder="••••••••"
                value={confirm}
                onChange={(e) => setConfirm(e.target.value)}
                required
              />
            </label>
            <button className="btn" disabled={busy}>
              {busy ? "Creating account..." : "Create account"}
            </button>
            <div className="login-help">
              Already have an account?{" "}
              <a href="#login" onClick={(e) => { e.preventDefault(); switchMode("login"); }}>
                Log in
              </a>
            </div>
          </form>
        )}
      </div>
    </div>
  );
}
