import React, { Component, ErrorInfo, ReactNode } from 'react';
import { analytics } from '@/utils/analytics';
import { errorHandler, ErrorType } from '@/lib/error-handler';

interface Props {
  children: ReactNode;
  fallback?: ReactNode;
}

interface State {
  hasError: boolean;
  error?: Error;
  errorInfo?: ErrorInfo;
  errorCount: number;
  customError?: any;
  retryCount: number;
  maxRetries: number;
}

export class ErrorBoundary extends Component<Props, State> {
  constructor(props: Props) {
    super(props);
    this.state = { 
      hasError: false, 
      errorCount: 0,
      retryCount: 0,
      maxRetries: 3
    };
  }

  static getDerivedStateFromError(error: Error): State {
    return { 
      hasError: true, 
      error, 
      errorCount: 0,
      retryCount: 0,
      maxRetries: 3
    };
  }

  componentDidCatch(error: Error, errorInfo: ErrorInfo) {
    const customError = errorHandler.createError(ErrorType.CLASSIFICATION_ERROR, error.message, {
      userMessage: 'Something went wrong while processing your request. Please try again.',
      context: {
        componentStack: errorInfo.componentStack,
        errorBoundary: true,
        errorCount: this.state.errorCount + 1,
      },
    });

    this.setState((prevState) => ({
      error,
      errorInfo,
      customError,
      errorCount: prevState.errorCount + 1,
    }));

    // Track error in analytics
    analytics.trackError(error, errorInfo, true);

    // Log error to console in development
    if (process.env.NODE_ENV === 'development') {
      console.error('ErrorBoundary caught an error:', error, errorInfo);
    }

    // Track repeated errors
    if (this.state.errorCount > 2) {
      analytics.event({
        action: 'repeated_error',
        category: 'Error',
        label: error.message,
        value: this.state.errorCount,
      });
    }
  }

  handleRetry = () => {
    if (this.state.retryCount >= this.state.maxRetries) {
      // Max retries reached, suggest page reload
      return;
    }

    analytics.event({
      action: 'error_retry',
      category: 'User_Interaction',
      label: this.state.error?.message || 'unknown',
      value: this.state.retryCount + 1,
    });

    this.setState((prevState) => ({
      hasError: false, 
      error: undefined, 
      errorInfo: undefined,
      customError: undefined,
      retryCount: prevState.retryCount + 1,
    }));
  };

  canRetry = () => {
    return this.state.retryCount < this.state.maxRetries;
  };

  render() {
    if (this.state.hasError) {
      if (this.props.fallback) {
        return this.props.fallback;
      }

      return (
        <div className="min-h-screen flex items-center justify-center p-4 sm:p-6 bg-gray-50 dark:bg-gray-900">
          <div className="max-w-md w-full bg-white dark:bg-gray-800 rounded-lg shadow-lg p-6 sm:p-8 text-center">
            <div className="mb-4">
              <div className="w-16 h-16 bg-red-100 dark:bg-red-900 rounded-full flex items-center justify-center mx-auto mb-4">
                <svg
                  className="w-8 h-8 text-red-600 dark:text-red-400"
                  fill="none"
                  stroke="currentColor"
                  viewBox="0 0 24 24"
                >
                  <path
                    strokeLinecap="round"
                    strokeLinejoin="round"
                    strokeWidth={2}
                    d="M12 9v2m0 4h.01m-6.938 4h13.856c1.54 0 2.502-1.667 1.732-3L13.732 4c-.77-1.333-2.694-1.333-3.464 0L3.34 16c-.77 1.333.192 3 1.732 3z"
                  />
                </svg>
              </div>
              <h2 className="text-xl sm:text-2xl font-bold text-gray-900 dark:text-white mb-2">
                Oops! Something went wrong
              </h2>
              <p className="text-gray-600 dark:text-gray-300 mb-6 text-sm sm:text-base">
                {this.state.customError?.userMessage || 
                 "We're sorry, but something unexpected happened. The error has been logged and we'll look into it."}
              </p>
              
              {!this.canRetry() && (
                <div className="mb-4 p-3 bg-yellow-50 dark:bg-yellow-900/20 border border-yellow-200 dark:border-yellow-800 rounded-lg">
                  <p className="text-sm text-yellow-800 dark:text-yellow-200">
                    Multiple retry attempts failed. Please try reloading the page.
                  </p>
                </div>
              )}
            </div>

            <div className="space-y-3">
              {this.canRetry() && (
                <button
                  onClick={this.handleRetry}
                  className="w-full bg-accent hover:bg-accent/90 text-white font-medium py-2 px-4 rounded-lg transition-colors focus:outline-none focus:ring-2 focus:ring-accent focus:ring-offset-2"
                >
                  Try Again {this.state.retryCount > 0 && `(${this.state.retryCount}/${this.state.maxRetries})`}
                </button>
              )}
              
              <button
                onClick={() => window.location.reload()}
                className="w-full bg-gray-200 hover:bg-gray-300 dark:bg-gray-700 dark:hover:bg-gray-600 text-gray-800 dark:text-white font-medium py-2 px-4 rounded-lg transition-colors focus:outline-none focus:ring-2 focus:ring-gray-500 focus:ring-offset-2"
              >
                Reload Page
              </button>
              
              <button
                onClick={() => window.history.back()}
                className="w-full text-accent hover:text-accent/80 font-medium py-2 px-4 rounded-lg transition-colors focus:outline-none focus:ring-2 focus:ring-accent focus:ring-offset-2"
              >
                Go Back
              </button>
            </div>

            {process.env.NODE_ENV === 'development' && this.state.error && (
              <details className="mt-6 text-left">
                <summary className="cursor-pointer text-sm font-medium text-gray-700 dark:text-gray-300 mb-2 hover:text-gray-900 dark:hover:text-white">
                  Error Details (Development Only)
                </summary>
                <div className="mt-2 p-3 bg-gray-100 dark:bg-gray-700 rounded text-xs font-mono text-gray-800 dark:text-gray-200 overflow-auto max-h-40">
                  <div className="font-semibold mb-2 text-red-600 dark:text-red-400">Error:</div>
                  <div className="mb-3">{this.state.error.toString()}</div>
                  
                  {this.state.customError && (
                    <>
                      <div className="font-semibold mb-2">Error Type:</div>
                      <div className="mb-3">{this.state.customError.type}</div>
                      <div className="font-semibold mb-2">User Message:</div>
                      <div className="mb-3">{this.state.customError.userMessage}</div>
                    </>
                  )}
                  
                  {this.state.errorInfo && (
                    <>
                      <div className="font-semibold mb-2">Component Stack:</div>
                      <pre className="whitespace-pre-wrap">
                        {this.state.errorInfo.componentStack}
                      </pre>
                    </>
                  )}
                  
                  <div className="font-semibold mb-2 mt-3">Retry Info:</div>
                  <div>Attempts: {this.state.retryCount}/{this.state.maxRetries}</div>
                  <div>Error Count: {this.state.errorCount}</div>
                </div>
              </details>
            )}
          </div>
        </div>
      );
    }

    return this.props.children;
  }
}
