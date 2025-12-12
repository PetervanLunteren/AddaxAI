/**
 * Main App component.
 *
 * Following DEVELOPERS.md principles:
 * - Simple, clear structure
 * - Type hints everywhere
 */

import { QueryClientProvider } from "@tanstack/react-query";
import { queryClient } from "./lib/query-client";

function App() {
  return (
    <QueryClientProvider client={queryClient}>
      <div className="min-h-screen bg-gray-50">
        <header className="bg-white shadow">
          <div className="mx-auto max-w-7xl px-4 py-6 sm:px-6 lg:px-8">
            <h1 className="text-3xl font-bold tracking-tight text-gray-900">
              AddaxAI
            </h1>
          </div>
        </header>
        <main>
          <div className="mx-auto max-w-7xl px-4 py-6 sm:px-6 lg:px-8">
            <div className="rounded-lg bg-white p-8 shadow">
              <h2 className="text-2xl font-semibold text-gray-900">
                Camera Trap Wildlife Analysis Platform
              </h2>
              <p className="mt-4 text-gray-600">
                Frontend initialized successfully. Backend API available at{" "}
                <code className="rounded bg-gray-100 px-2 py-1 text-sm">
                  http://127.0.0.1:8000
                </code>
              </p>
              <div className="mt-6">
                <p className="text-sm text-gray-500">
                  ✅ React + TypeScript + Vite
                  <br />
                  ✅ TanStack Query for server state
                  <br />
                  ✅ Tailwind CSS for styling
                  <br />
                  ✅ API client configured
                </p>
              </div>
            </div>
          </div>
        </main>
      </div>
    </QueryClientProvider>
  );
}

export default App;
