import ChromosomeTrack3D from "./ChromosomeTrack3D";
import TrackMappingDemo from "./TrackMappingDemo";

function App() {
  const params = new URLSearchParams(window.location.search);
  const view = params.get("view");
  
  if (view === "demo") {
    return <TrackMappingDemo />;
  }
  
  return <ChromosomeTrack3D />;
}

export default App;
