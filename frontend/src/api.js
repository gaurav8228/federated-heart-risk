// ---------------------------------------------------------------------------
// api.js - the ONE place that talks to the FastAPI backend.
//
// The JWT is saved in the browser's localStorage after login and attached to
// every request as "Authorization: Bearer <token>".
// ---------------------------------------------------------------------------

const API_BASE = "http://localhost:8000"; // change here if the backend moves

// --- tiny helpers for the saved login --------------------------------------
export function getToken() {
  return localStorage.getItem("token");
}
export function getRole() {
  return localStorage.getItem("role") || "doctor";
}
export function getName() {
  return localStorage.getItem("name") || "";
}
export function saveLogin(data) {
  // data = { access_token, role, name } from POST /api/auth/login
  localStorage.setItem("token", data.access_token);
  localStorage.setItem("role", data.role);
  localStorage.setItem("name", data.name);
}
export function logout() {
  localStorage.clear();
  window.location.href = "/login";
}

// --- one fetch wrapper for every API call ----------------------------------
export async function apiFetch(path, options = {}) {
  const token = getToken();
  const res = await fetch(API_BASE + path, {
    ...options,
    headers: {
      "Content-Type": "application/json",
      ...(token ? { Authorization: `Bearer ${token}` } : {}),
      ...(options.headers || {}),
    },
  });

  // 401 = token missing/expired -> go back to the login page
  if (res.status === 401) {
    logout();
    throw new Error("Your session expired. Please log in again.");
  }

  const data = await res.json().catch(() => ({}));
  if (!res.ok) {
    // FastAPI puts a friendly message in "detail"
    const msg = Array.isArray(data.detail) ? data.detail[0]?.msg : data.detail;
    throw new Error(msg || `Request failed (HTTP ${res.status}).`);
  }
  return data;
}

export { API_BASE };
