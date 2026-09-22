import { useState, useRef, useEffect } from 'react';
import { useQuery, useMutation } from '@tanstack/react-query';
import { fetchSymptoms, predictDisease } from './api';
import { Search, X, Activity, AlertTriangle, ChevronRight, Loader2 } from 'lucide-react';
import { clsx } from 'clsx';
import { twMerge } from 'tailwind-merge';

function cn(...inputs: (string | undefined | null | false)[]) {
  return twMerge(clsx(inputs));
}

export const DiseasePredictor = () => {
  const [selectedSymptoms, setSelectedSymptoms] = useState<string[]>([]);
  const [searchTerm, setSearchTerm] = useState('');
  const [isDropdownOpen, setIsDropdownOpen] = useState(false);
  const dropdownRef = useRef<HTMLDivElement>(null);

  const { data: allSymptoms = [], isLoading: isLoadingSymptoms } = useQuery({
    queryKey: ['symptoms'],
    queryFn: fetchSymptoms
  });

  const { mutate: getPrediction, data: predictions, isPending, error } = useMutation({
    mutationFn: predictDisease
  });

  useEffect(() => {
    const handleClickOutside = (event: MouseEvent) => {
      if (dropdownRef.current && !dropdownRef.current.contains(event.target as Node)) {
        setIsDropdownOpen(false);
      }
    };
    document.addEventListener('mousedown', handleClickOutside);
    return () => document.removeEventListener('mousedown', handleClickOutside);
  }, []);

  const availableSymptoms = allSymptoms.filter(s => 
    !selectedSymptoms.includes(s) && 
    s.replace(/_/g, ' ').toLowerCase().includes(searchTerm.toLowerCase())
  );

  const handleSelect = (symptom: string) => {
    setSelectedSymptoms(prev => [...prev, symptom]);
    setSearchTerm('');
    setIsDropdownOpen(false);
  };

  const handleRemove = (symptom: string) => {
    setSelectedSymptoms(prev => prev.filter(s => s !== symptom));
  };

  const handlePredict = () => {
    if (selectedSymptoms.length > 0) {
      getPrediction(selectedSymptoms);
    }
  };

  const formatSymptom = (str: string) => {
    return str.split('_').map(word => word.charAt(0).toUpperCase() + word.slice(1)).join(' ');
  };

  return (
    <div className="max-w-4xl mx-auto space-y-8 animate-in fade-in slide-in-from-bottom-4 duration-700">
      
      {/* Disclaimer */}
      <div className="bg-amber-500/10 border border-amber-500/30 rounded-lg p-4 flex items-start gap-3">
        <AlertTriangle className="w-6 h-6 text-amber-500 shrink-0 mt-0.5" />
        <div>
          <h3 className="font-semibold text-amber-500">Academic Demo</h3>
          <p className="text-amber-200/80 text-sm mt-1">
            This is an academic demo, not a medical diagnosis tool. Consult a qualified doctor for real symptoms. 
            The dataset uses synthesized rules and may not represent clinical reality.
          </p>
        </div>
      </div>

      <div className="grid lg:grid-cols-2 gap-8">
        
        {/* Input Section */}
        <div className="space-y-6">
          <div className="glass-panel p-6">
            <h2 className="text-xl font-bold mb-4 flex items-center gap-2">
              <Activity className="text-indigo-400" />
              Symptom Checker
            </h2>
            
            <div className="relative" ref={dropdownRef}>
              <div className="relative">
                <Search className="absolute left-3 top-1/2 -translate-y-1/2 w-5 h-5 text-slate-400" />
                <input
                  type="text"
                  placeholder={isLoadingSymptoms ? "Loading symptoms..." : "Search for a symptom..."}
                  className="w-full bg-slate-900/50 border border-slate-700 rounded-lg pl-10 pr-4 py-3 focus:outline-none focus:ring-2 focus:ring-indigo-500 transition-shadow"
                  value={searchTerm}
                  onChange={(e) => {
                    setSearchTerm(e.target.value);
                    setIsDropdownOpen(true);
                  }}
                  onFocus={() => setIsDropdownOpen(true)}
                  disabled={isLoadingSymptoms}
                />
              </div>

              {isDropdownOpen && searchTerm.length > 0 && (
                <div className="absolute z-10 w-full mt-2 glass-panel max-h-60 overflow-y-auto py-2 shadow-2xl">
                  {availableSymptoms.length > 0 ? (
                    availableSymptoms.map(symptom => (
                      <button
                        key={symptom}
                        onClick={() => handleSelect(symptom)}
                        className="w-full text-left px-4 py-2 hover:bg-indigo-500/20 text-slate-200 transition-colors"
                      >
                        {formatSymptom(symptom)}
                      </button>
                    ))
                  ) : (
                    <div className="px-4 py-2 text-slate-400">No matching symptoms found</div>
                  )}
                </div>
              )}
            </div>

            <div className="mt-6">
              <h3 className="text-sm font-medium text-slate-400 mb-3">Selected Symptoms ({selectedSymptoms.length})</h3>
              <div className="flex flex-wrap gap-2 mb-6">
                {selectedSymptoms.map(symptom => (
                  <span 
                    key={symptom}
                    className="inline-flex items-center gap-1.5 px-3 py-1.5 bg-indigo-500/20 text-indigo-200 border border-indigo-500/30 rounded-full text-sm group"
                  >
                    {formatSymptom(symptom)}
                    <button 
                      onClick={() => handleRemove(symptom)}
                      className="hover:bg-indigo-500/40 rounded-full p-0.5 transition-colors"
                    >
                      <X className="w-3.5 h-3.5" />
                    </button>
                  </span>
                ))}
                {selectedSymptoms.length === 0 && (
                  <p className="text-slate-500 text-sm italic">No symptoms selected yet.</p>
                )}
              </div>

              <button
                onClick={handlePredict}
                disabled={selectedSymptoms.length === 0 || isPending}
                className="w-full btn-primary flex items-center justify-center gap-2 py-3"
              >
                {isPending ? (
                  <>
                    <Loader2 className="w-5 h-5 animate-spin" />
                    Analyzing...
                  </>
                ) : (
                  <>
                    Predict Disease
                    <ChevronRight className="w-5 h-5" />
                  </>
                )}
              </button>
              
              {error && (
                <p className="text-red-400 text-sm mt-3 text-center">
                  {(error as any)?.response?.data?.detail || "An error occurred during prediction."}
                </p>
              )}
            </div>
          </div>
        </div>

        {/* Results Section */}
        <div className="space-y-4">
          <h2 className="text-xl font-bold flex items-center gap-2 mb-2">
            Predicted Results
          </h2>
          
          {predictions ? (
            <div className="space-y-4">
              {predictions.map((pred, idx) => (
                <div 
                  key={pred.disease} 
                  className={cn(
                    "glass-panel p-5 relative overflow-hidden transition-all duration-500 hover:scale-[1.02]",
                    idx === 0 ? "border-indigo-500/50 shadow-indigo-500/10" : ""
                  )}
                  style={{ animationDelay: `${idx * 150}ms` }}
                >
                  {/* Progress bar background */}
                  <div 
                    className="absolute inset-0 bg-gradient-to-r from-indigo-600/10 to-transparent transition-all duration-1000"
                    style={{ width: `${pred.confidence}%` }}
                  />
                  
                  <div className="relative z-10 flex justify-between items-start mb-2">
                    <h3 className={cn("font-bold text-lg", idx === 0 ? "text-indigo-300" : "text-slate-200")}>
                      {pred.disease}
                    </h3>
                    <span className="font-mono text-lg text-slate-300">
                      {pred.confidence.toFixed(1)}%
                    </span>
                  </div>
                  
                  <div className="relative z-10 w-full bg-slate-900/50 rounded-full h-2 mb-4 overflow-hidden">
                    <div 
                      className={cn("h-full rounded-full transition-all duration-1000", idx === 0 ? "bg-indigo-500" : "bg-slate-600")}
                      style={{ width: `${pred.confidence}%` }}
                    />
                  </div>

                  {pred.contributing_symptoms.length > 0 && (
                    <div className="relative z-10 mt-3">
                      <p className="text-xs text-slate-400 mb-1.5 uppercase tracking-wider font-semibold">Key Contributing Symptoms:</p>
                      <div className="flex flex-wrap gap-1.5">
                        {pred.contributing_symptoms.map(s => (
                          <span key={s} className="text-xs px-2 py-1 bg-slate-800 rounded-md border border-slate-700 text-slate-300">
                            {formatSymptom(s)}
                          </span>
                        ))}
                      </div>
                    </div>
                  )}
                </div>
              ))}
            </div>
          ) : (
            <div className="glass-panel p-10 flex flex-col items-center justify-center text-slate-500 h-[400px]">
              <Activity className="w-12 h-12 mb-4 opacity-20" />
              <p>Select symptoms and click predict to see results</p>
            </div>
          )}
        </div>
      </div>
    </div>
  );
};
