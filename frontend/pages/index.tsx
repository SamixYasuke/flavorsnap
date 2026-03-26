import { useRef, useState } from "react";
import { api } from "@/utils/api";
import { ErrorMessage } from "@/components/ErrorMessage";
import { useTranslation } from "next-i18next";
import { serverSideTranslations } from "next-i18next/serverSideTranslations";
import type { GetStaticProps } from "next";
import LanguageSwitcher from "@/components/LanguageSwitcher";
import { useKeyboardShortcuts } from "@/utils/useKeyboardShortcuts";


  const handleReset = () => {
    setImage(null);
    setClassification(null);
    setError(null);
    if (fileInputRef.current) {
      fileInputRef.current.value = "";
    }
  };

  const handleImageChange = (event: React.ChangeEvent<HTMLInputElement>) => {
    const file = event.target.files?.[0];
    if (file) {
      const imageUrl = URL.createObjectURL(file);
      setImage(imageUrl);
      setError(null);
      setClassification(null);
    }
  };

  const handleClassify = async () => {
    if (!image) return;

    setLoading(true);
    setError(null);

    // Announce to screen readers that classification is starting
    const announcement = document.getElementById('classification-announcement');
    if (announcement) {
      announcement.textContent = t('classifying');
    }

    try {
      // Example API call with error handling
      const response = await api.post('/api/classify', {
        image: image
      }, {
        retries: 2,
        retryDelay: 1000
      });

      if (response.error) {
        setError(response.error);
      } else if (response.data) {
        setClassification(response.data);
      }
    } catch (err: any) {
      setError(t('error_classify_retry'));
      console.error('Classification error:', err);
    } finally {
      setLoading(false);
    }
  };

  const handleOpenPicker = () => {
    fileInputRef.current?.click();
  };

  useKeyboardShortcuts([
    { key: 'o', action: handleOpenPicker },
    { key: 'c', action: () => image && !loading && handleClassify() },
    { key: 'r', action: handleReset },
    { key: 'Escape', action: handleReset },
  ]);

  return (
    <div className="min-h-screen flex flex-col items-center justify-center p-6 bg-gray-50 dark:bg-gray-900 transition-colors duration-300">
      <div className="absolute top-4 end-4">
        <LanguageSwitcher />
      </div>

      <header className="text-center mb-8">
        <h1 className="text-4xl font-extrabold mb-2 text-gray-900 dark:text-white">
          {t("snap_your_food")} 🍛
        </h1>
        <p className="text-gray-600 dark:text-gray-400">
          {t("shortcut_hint", "Press 'O' to open camera, 'C' to classify, 'R' to reset")}
        </p>
      </header>

      {/* Screen reader announcements */}
      <div
        id="classification-announcement"
        role="status"
        aria-live="polite"
        className="sr-only"
      />

      <div
        id="error-announcement"
        role="alert"
        aria-live="assertive"
        className="sr-only"
      />

      <input
        type="file"
        accept="image/*"
        capture="environment"
        ref={fileInputRef}
        onChange={handleImageChange}
        className="hidden"
        aria-label={t("select_image_file")}
      />

      {!image && (
        <button
          onClick={handleOpenPicker}
          onKeyPress={(e) => {
            if (e.key === 'Enter' || e.key === ' ') {
              e.preventDefault();
              handleOpenPicker();
            }
          }}
          className="group relative bg-indigo-600 text-white px-8 py-4 rounded-2xl shadow-lg hover:bg-indigo-700 active:scale-95 transition-all focus:outline-none focus:ring-4 focus:ring-indigo-500/50"
          aria-label={t("open_camera")}
        >
          <span className="flex items-center gap-2 text-lg font-semibold">
            {t("open_camera")}
            <kbd className="hidden sm:inline-block px-2 py-0.5 text-xs bg-indigo-500 rounded border border-indigo-400">O</kbd>
          </span>
        </button>
      )}

      {error && (
        <div className="w-full max-w-sm mb-6">
          <ErrorMessage
            message={error}
            onRetry={() => handleClassify()}
            onDismiss={() => setError(null)}
          />
        </div>
      )}

      {image && (
        <div className="w-full max-w-md animate-fade-in" role="region" aria-label={t("image_preview")}>
          <div className="relative group">
            <img
              src={image}
              alt={t("preview_alt")}
              className="rounded-3xl shadow-2xl w-full h-auto object-cover border-4 border-white dark:border-gray-800"
            />
            <button
              onClick={handleReset}
              className="absolute top-4 right-4 bg-red-500 text-white p-2 rounded-full shadow-lg hover:bg-red-600 transition-colors focus:outline-none focus:ring-2 focus:ring-red-500 focus:ring-offset-2"
              title={t("clear_image", "Clear Image (R)")}
            >
              <svg xmlns="http://www.w3.org/2000/svg" className="h-6 w-6" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M6 18L18 6M6 6l12 12" />
              </svg>
            </button>
          </div>

          <div className="mt-8 flex flex-col gap-4">
            <button
              onClick={handleClassify}
              disabled={loading}
              className="w-full flex items-center justify-center gap-2 bg-emerald-600 text-white px-8 py-4 rounded-2xl shadow-lg hover:bg-emerald-700 disabled:bg-gray-400 disabled:cursor-not-allowed transition-all active:scale-95 focus:outline-none focus:ring-4 focus:ring-emerald-500/50"
              aria-label={loading ? t('classifying') : t('classify_food')}
            >
              <span className="text-lg font-bold">
                {loading ? t('classifying') : t('classify_food')}
              </span>
              {!loading && <kbd className="hidden sm:inline-block px-2 py-0.5 text-xs bg-emerald-500 rounded border border-emerald-400 uppercase">C</kbd>}
              {loading && (
                <svg className="animate-spin h-5 w-5 text-white" xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24">
                  <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4"></circle>
                  <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"></path>
                </svg>
              )}
            </button>
            
            <button
              onClick={handleReset}
              className="w-full py-2 text-gray-500 dark:text-gray-400 hover:text-gray-800 dark:hover:text-white transition-colors text-sm font-medium"
            >
              {t("reset", "Reset")} <kbd className="ml-1 uppercase">R</kbd>
            </button>
          </div>

          {classification && (
            <div
              className="mt-8 p-6 bg-white dark:bg-gray-800 rounded-3xl shadow-xl border border-gray-100 dark:border-gray-700 animate-slide-up"
              role="region"
              aria-label={t('classification_result')}
            >
              <h3 className="text-xl font-bold text-gray-900 dark:text-white mb-4 flex items-center gap-2">
                <span className="text-emerald-500">✨</span> {t('classification_result')}
              </h3>
              <div className="bg-gray-50 dark:bg-gray-900 p-4 rounded-xl overflow-auto border border-gray-100 dark:border-gray-700">
                <pre className="text-sm text-gray-800 dark:text-gray-200 font-mono">
                  {JSON.stringify(classification, null, 2)}
                </pre>
              </div>
            </div>
          )}
        </div>
      )}
    </div>
  );
}

export const getStaticProps: GetStaticProps = async ({ locale }) => ({
  props: {
    ...(await serverSideTranslations(locale ?? "en", ["common"])),
  },
});