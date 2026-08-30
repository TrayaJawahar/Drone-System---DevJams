import { BrowserRouter, Routes, Route } from "react-router-dom";
import { IndexPage } from "./pages/IndexPage";
import { MissionPage } from "./pages/MissionPage";

function App() {
  return (
    <BrowserRouter>
      <Routes>
        <Route path="/" element={<IndexPage />} />
        <Route path="/mission" element={<MissionPage />} />
      </Routes>
    </BrowserRouter>
  );
}

export default App;
