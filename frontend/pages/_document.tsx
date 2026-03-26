import { Html, Head, Main, NextScript } from "next/document";
import type { DocumentProps } from "next/document";

const rtlLocales = ["ar"];

export default function Document(props: DocumentProps) {
  const locale = props.__NEXT_DATA__?.locale || "en";
  const dir = rtlLocales.includes(locale) ? "rtl" : "ltr";

  return (

      </Head>
      <body className="antialiased">
        <div id="skip-to-content">
          <a href="#main-content" className="sr-only focus:not-sr-only focus:absolute focus:top-4 focus:left-4 bg-blue-600 text-white px-4 py-2 rounded-md z-50">
            Skip to main content
          </a>
        </div>
        <Main />
        <NextScript />
      </body>
    </Html>
  );
}
