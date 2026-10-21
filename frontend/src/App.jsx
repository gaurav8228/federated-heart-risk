// App.jsx - the routes (pages) of the application.
// Every page except /login requires a token, otherwise we send the user
// back to the login page.
import { BrowserRouter, Navigate, Route, Routes } from "react-router-dom";
import { getToken } from "./api";
import Layout from "./components/Layout";
import Dashboard from "./pages/Dashboard";
import Predict from "./pages/Predict";
import History from "./pages/History";
import PredictionDetail from "./pages/PredictionDetail";
import FlTraining from "./pages/FlTraining";
import Admin from "./pages/Admin";
import Settings from "./pages/Settings";
import Login from "./pages/Login";

// Protects private pages: no token -> redirect to /login
function Protected({ children }) {
  if (!getToken()) return <Navigate to="/login" replace />;
  return children;
}

export default function App() {
  return (
    <BrowserRouter>
      <Routes>
        <Route path="/login" element={<Login />} />

        {/* All these pages share the sidebar layout */}
        <Route
          element={
            <Protected>
              <Layout />
            </Protected>
          }
        >
          <Route path="/" element={<Dashboard />} />
          <Route path="/predict" element={<Predict />} />
          <Route path="/history" element={<History />} />
          <Route path="/history/:id" element={<PredictionDetail />} />
          <Route path="/fl-training" element={<FlTraining />} />
          <Route path="/admin" element={<Admin />} />
          <Route path="/settings" element={<Settings />} />
        </Route>

        <Route path="*" element={<Navigate to="/" replace />} />
      </Routes>
    </BrowserRouter>
  );
}
