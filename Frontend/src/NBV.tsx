import "./NBV.css";

export interface NBVProps {
  onClose?: () => void;
  onMinimize?: () => void;
  minimized?: boolean;
}

export default function NBV({ onClose, onMinimize, minimized }: NBVProps) {
  return (
    <div className="nbv">
      <div
        className={`nbv__ribbon ${minimized ? "nbv__ribbon--minimized" : ""}`}
      >
        <button type="button" className="nbv__btn nbv__btn--title" title="NBV">
          NBV
        </button>
        <div className="nbv__spacer" />
        {onMinimize && (
          <button
            type="button"
            className="nbv__btn"
            onClick={onMinimize}
            title="Minimize"
          >
            {minimized ? "▢" : "−"}
          </button>
        )}
        {onClose && (
          <button
            type="button"
            className="nbv__btn"
            onClick={onClose}
            title="Close"
          >
            ✕
          </button>
        )}
      </div>
    </div>
  );
}
