import { useEffect, useRef, useState } from "react";

/**
 * Loading / error / empty state for one request.
 *
 * The loader is kept in a ref so a component can pass an inline arrow function without
 * restarting the request on every render; `deps` decides when it actually re-runs.
 * A cancelled flag drops responses from a superseded request, which is what stops a
 * slow first load from overwriting a newer one.
 */
export function useApi(loader, deps = []) {
  const [state, setState] = useState({ status: "loading", data: null, error: null });
  const loaderRef = useRef(loader);
  loaderRef.current = loader;

  useEffect(() => {
    let cancelled = false;
    setState((current) => ({
      status: "loading",
      data: current.data,
      error: null,
    }));

    loaderRef
      .current()
      .then((data) => {
        if (!cancelled) setState({ status: "ready", data, error: null });
      })
      .catch((error) => {
        if (!cancelled) setState({ status: "error", data: null, error });
      });

    return () => {
      cancelled = true;
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, deps);

  return state;
}
