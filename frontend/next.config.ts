import type { NextConfig } from "next";
import { i18n } from "./next-i18next.config";

// Note: next-pwa is required for this feature, but currently we are focusing on fixing
// the cache invalidation issues. Using StaleWhileRevalidate for images ensures
// they are updated when the underlying data changes.
const withPWA = require("next-pwa")({
  dest: "public",
  disable: process.env.NODE_ENV === "development",
  register: true,
  skipWaiting: true,
  runtimeCaching: [
    {
      urlPattern: /^https?.*\/api\/.*$/,
      handler: "NetworkFirst",
      options: {
        cacheName: "api-cache",
        expiration: {
          maxEntries: 50,
          maxAgeSeconds: 60 * 60 * 24, // 24 hours
        },
        networkTimeoutSeconds: 10, // Ensure we don't wait forever on flaky networks
      },
    },
    {
      urlPattern: /\.(?:jpg|jpeg|png|gif|webp|svg)$/,
      handler: "StaleWhileRevalidate", // Changed from CacheFirst to fix invalidation issues
      options: {
        cacheName: "image-cache",
        expiration: {
          maxEntries: 100,
          maxAgeSeconds: 60 * 60 * 24 * 30, // 30 days
        },
      },
    },
    {
      urlPattern: /\.(?:js|css|html)$/,
      handler: "StaleWhileRevalidate",
      options: {
        cacheName: "static-resources",
      },
    },
  ],
});
// 1. Properly import the i18n configuration
const { i18n } = require("./next-i18next.config.js");

const nextConfig: NextConfig = {
  reactStrictMode: true,
  i18n,
};

export default withPWA(nextConfig);
