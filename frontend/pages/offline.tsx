import React from 'react';
import Head from 'next/head';

export default function Offline() {
  return (
    <>
      <Head>
        <title>Offline - FlavorSnap</title>
        <meta name="description" content="You are currently offline" />
        <meta name="viewport" content="width=device-width, initial-scale=1" />
        <meta name="robots" content="noindex" />
      </Head>
      
      <div className="min-h-screen flex items-center justify-center bg-gray-50">
        <div className="max-w-md w-full bg-white shadow-lg rounded-lg p-8 text-center">
          <div className="mb-6">
            <div className="w-20 h-20 bg-amber-100 rounded-full flex items-center justify-center mx-auto mb-4">
              <svg className="w-10 h-10 text-amber-600" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M8.111 16.404a5.5 5.5 0 017.778 0M12 20h.01m-7.08-7.071c3.904-3.905 10.236-3.905 14.141 0M1.394 9.393c5.857-5.857 15.355-5.857 21.213 0" />
              </svg>
            </div>
            <h1 className="text-2xl font-bold text-gray-900 mb-2">You're Offline</h1>
            <p className="text-gray-600 mb-6">
              It looks like you've lost your internet connection. Some features may not be available until you're back online.
            </p>
          </div>
          
          <div className="space-y-3">
            <div className="bg-amber-50 border border-amber-200 rounded-lg p-4">
              <h3 className="font-semibold text-amber-800 mb-2">Available Offline:</h3>
              <ul className="text-sm text-amber-700 space-y-1">
                <li>• Previously viewed food images</li>
                <li>• Cached nutrition data</li>
                <li>• App settings and preferences</li>
              </ul>
            </div>
            
            <button 
              onClick={() => window.location.reload()}
              className="w-full bg-amber-600 text-white py-3 px-4 rounded-lg hover:bg-amber-700 transition duration-200 font-medium"
            >
              Try Again
            </button>
          </div>
        </div>
      </div>
    </>
  );
}
