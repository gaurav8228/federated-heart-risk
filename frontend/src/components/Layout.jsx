// Layout: dark sidebar with the doctor's name on top + the current page.
import { NavLink, Outlet, useNavigate } from "react-router-dom";
import { getRole, getName, logout } from "../api";
import Icon, { HeartLogo } from "./Icon";

export default function Layout() {
  const role = getRole();
  const name = getName();
  const navigate = useNavigate();

  const links = [
    { to: "/", label: "Dashboard", icon: "dashboard" },
    { to: "/predict", label: "New Prediction", icon: "predict" },
    { to: "/history", label: "History", icon: "history" },
    { to: "/fl-training", label: "FL Training", icon: "chart" },
    { to: "/settings", label: "Settings", icon: "settings" },
  ];
  if (role === "admin") {
    links.push({ to: "/admin", label: "Admin", icon: "admin" });
  }

  return (
    <div className="app-shell">
      <aside className="sidebar">
        <div className="sidebar-brand">
          <HeartLogo size={30} /> HeartRisk AI
        </div>

        <div className="sidebar-user">
          <div className="name">{name}</div>
          <div className="role">{role}</div>
        </div>

        {links.map((l) => (
          <NavLink
            key={l.to}
            to={l.to}
            end={l.to === "/"}
            className={({ isActive }) => "nav-link" + (isActive ? " active" : "")}
          >
            <Icon name={l.icon} /> {l.label}
          </NavLink>
        ))}

        <button
          className="btn btn-outline"
          style={{ marginTop: "auto" }}
          onClick={() => {
            logout();
            navigate("/login");
          }}
        >
          Log out
        </button>
      </aside>

      <main className="main">
        <Outlet />
      </main>
    </div>
  );
}
