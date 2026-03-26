interface ApiResponse<T = any> {
  data?: T;
  error?: string;
  status: number;
}

interface ApiOptions extends RequestInit {
  retries?: number;
  retryDelay?: number;
}

class ApiError extends Error {
  constructor(
    message: string,
    public status: number,
    public data?: any,
  ) {
    super(message);
    this.name = "ApiError";
  }
}

const sleep = (ms: number) => new Promise((resolve) => setTimeout(resolve, ms));

const apiRequest = async <T = any>(
  url: string,
  options: ApiOptions = {},
): Promise<ApiResponse<T>> => {
  const { retries = 3, retryDelay = 1000, ...fetchOptions } = options;

  let lastError: Error | null = null;

  for (let attempt = 0; attempt <= retries; attempt++) {
    try {
      const isFormData = typeof FormData !== "undefined" && fetchOptions.body instanceof FormData;
      const defaultHeaders = isFormData ? {} : { "Content-Type": "application/json" };

      const response = await fetch(url, {
        headers: {
          ...defaultHeaders,
          ...fetchOptions.headers,
        },
        ...fetchOptions,
      });

      const data = await response.json().catch(() => null);

      if (!response.ok) {
        const errorMessage =
          data?.error || data?.message || `HTTP ${response.status}`;
        throw new ApiError(errorMessage, response.status, data);
      }

      return {
        data,
        status: response.status,
      };
    } catch (error) {
      lastError =
        error instanceof Error ? error : new Error("Unknown error occurred");

      // Don't retry on client errors (4xx) except for 429 (rate limit)
      if (
        lastError instanceof ApiError &&
        lastError.status >= 400 &&
        lastError.status < 500 &&
        lastError.status !== 429
      ) {
        break;
      }

      // If this is the last attempt, don't wait
      if (attempt < retries) {
        await sleep(retryDelay * Math.pow(2, attempt)); // Exponential backoff
      }
    }
  }

  return {
    error: lastError?.message || "Request failed",
    status: lastError instanceof ApiError ? lastError.status : 500,
  };
};

// API methods with error handling
export const api = {
  get: <T = any>(url: string, options?: ApiOptions) =>
    apiRequest<T>(url, { method: "GET", ...options }),

  post: <T = any>(url: string, data?: any, options?: ApiOptions) =>
    apiRequest<T>(url, {
      method: "POST",
      body: (typeof FormData !== "undefined" && data instanceof FormData) ? data : (data ? JSON.stringify(data) : undefined),
      ...options,
    }),

  put: <T = any>(url: string, data?: any, options?: ApiOptions) =>
    apiRequest<T>(url, {
      method: "PUT",
      body: data ? JSON.stringify(data) : undefined,
      ...options,
    }),

  delete: <T = any>(url: string, options?: ApiOptions) =>
    apiRequest<T>(url, { method: "DELETE", ...options }),
};

export { ApiError };
export type { ApiResponse };
