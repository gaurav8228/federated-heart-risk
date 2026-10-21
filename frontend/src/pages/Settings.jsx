// Settings: shows the logged-in profile (GET /api/auth/me) and a logout button.
import { useEffect, useState } from "react";
import { useNavigate } from "react-router-dom";
import { apiFetch, logout } from "../api";

export default function Settings() {
  const [user, setUser] = useState(null);
  const [error, setError] = useState("");
  const navigate = useNavigate();

  useEffect(() => {
    apiFetch("/api/auth/me")
      .then(setUser)
      .catch((err) => setError(err.message));
  }, []);

  return (
    <div>
      <h1>Settings</h1>

      {error && <div className="error-banner">{error}</div>}

      <div className="card" style={{ maxWidth: 480 }}>
        <h2>My Profile</h2>
        {user ? (
          <table>
            <tbody>
              <tr><td className="muted">Name</td><td>{user.name}</td></tr>
              <tr><td className="muted">Email</td><td>{user.email}</td></tr>
              <tr><td className="muted">Role</td><td style={{ textTransform: "capitalize" }}>{user.role}</td></tr>
              <tr><td className="muted">User ID</td><td>{user.id}</td></tr>
            </tbody>
          </table>
        ) : (
          <p>Loading...</p>
        )}
        <button
          className="btn btn-danger mt"
          onClick={() => {
            logout();
            navigate("/login");
          }}
        >
          Log out
        </button>
      </div>
    </div>
  );
}
