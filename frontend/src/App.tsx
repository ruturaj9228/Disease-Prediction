import React from 'react';
import { QueryClient, QueryClientProvider } from '@tanstack/react-query';
import { DiseasePredictor } from './DiseasePredictor';
import { Stethoscope } from 'lucide-react';

const queryClient = new QueryClient({
  defaultOptions: {
    queries: {
      refetchOnWindowFocus: false,
      staleTime: 1000 * 60 * 5, // 5 minutes
    },
  },
});

function App() {
  return (
    <QueryClientProvider client={queryClient}>
      <div className="min-h-screen relative overflow-hidden bg-slate-900 pb-20">
        {/* Background Gradients */}
        <div className="absolute top-[-10%] left-[-10%] w-96 h-96 bg-indigo-600/20 rounded-full blur-[100px] pointer-events-none" />
        <div className="absolute bottom-[-10%] right-[-10%] w-96 h-96 bg-emerald-600/10 rounded-full blur-[100px] pointer-events-none" />

        <div className="relative z-10 container mx-auto px-4 pt-12 pb-24">
          <header className="mb-12 text-center">
            <div className="inline-flex items-center justify-center p-3 bg-indigo-500/10 rounded-2xl mb-4 border border-indigo-500/20">
              <Stethoscope className="w-10 h-10 text-indigo-400" />
            </div>
            <h1 className="text-4xl md:text-5xl font-extrabold text-transparent bg-clip-text bg-gradient-to-r from-indigo-400 to-cyan-400 mb-4 tracking-tight">
              AI Disease Prediction
            </h1>
            <p className="text-slate-400 max-w-2xl mx-auto text-lg">
              Enter your symptoms to receive an AI-powered prediction of potential conditions, powered by a Random Forest machine learning model.
            </p>
          </header>

          <main>
            <DiseasePredictor />
          </main>
        </div>
      </div>
    </QueryClientProvider>
  );
}

export default App;
