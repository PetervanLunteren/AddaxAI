import type { Config } from "@docusaurus/types";
import type * as Preset from "@docusaurus/preset-classic";
import { themes as prismThemes } from "prism-react-renderer";

// AddaxAI documentation + landing site.
//
// Layout: a custom marketing homepage at "/" (src/pages/index.tsx) and the
// documentation under "/docs". Deployed to GitHub Pages at
// https://petervanlunteren.github.io/AddaxAI/ by
// .github/workflows/docs.yml on every push to main.
//
// If you later point a custom domain (e.g. docs.addaxdatascience.com) at the
// site, change `url` to that domain and `baseUrl` to "/", and drop a CNAME
// file into static/.
const config: Config = {
  title: "AddaxAI",
  tagline: "Simplifying camera trap image analysis with AI",
  favicon: "img/favicon.ico",

  future: {
    v4: true,
  },

  url: "https://petervanlunteren.github.io",
  baseUrl: "/AddaxAI/",

  organizationName: "PetervanLunteren",
  projectName: "AddaxAI",

  // Content has settled, so a dead link is a bug and should fail the build
  // rather than ship. Loosen to 'warn' only while restructuring.
  onBrokenLinks: "throw",
  onBrokenMarkdownLinks: "warn",

  i18n: {
    defaultLocale: "en",
    locales: ["en"],
  },

  markdown: {
    mermaid: true,
  },
  themes: [
    "@docusaurus/theme-mermaid",
    [
      // Offline, account-free search. Indexes the built docs at deploy time.
      "@easyops-cn/docusaurus-search-local",
      {
        hashed: true,
        indexBlog: false,
        docsRouteBasePath: "/docs",
        highlightSearchTermsOnTargetPage: true,
      },
    ],
  ],

  presets: [
    [
      "classic",
      {
        docs: {
          routeBasePath: "/docs",
          sidebarPath: "./sidebars.ts",
          // No editUrl on purpose. That adds an "Edit this page" link aimed at
          // people who would open a pull request, and this site is written for
          // end users. Typos and mistakes come in by email instead.
        },
        blog: false,
        theme: {
          customCss: "./src/css/custom.css",
        },
      } satisfies Preset.Options,
    ],
  ],

  themeConfig: {
    image: "img/logo-wordmark.png",
    colorMode: {
      defaultMode: "light",
      respectPrefersColorScheme: true,
    },
    navbar: {
      title: "AddaxAI",
      logo: {
        alt: "AddaxAI",
        src: "img/logo-mark.png",
      },
      items: [
        {
          type: "docSidebar",
          sidebarId: "docs",
          position: "left",
          label: "Documentation",
        },
        { to: "/docs/troubleshooting/faq", label: "FAQ", position: "left" },
        {
          href: "https://github.com/PetervanLunteren/AddaxAI/releases/latest",
          label: "Download",
          position: "right",
        },
        {
          href: "https://github.com/PetervanLunteren/AddaxAI",
          label: "GitHub",
          position: "right",
        },
      ],
    },
    footer: {
      style: "dark",
      links: [
        {
          title: "Docs",
          items: [
            { label: "Start here", to: "/docs/start-here/what-is-addaxai" },
            { label: "Understanding your results", to: "/docs/understanding/detections-events-observations" },
            { label: "Export columns", to: "/docs/reference/export-columns" },
            { label: "FAQ", to: "/docs/troubleshooting/faq" },
          ],
        },
        {
          title: "More",
          items: [
            {
              label: "GitHub",
              href: "https://github.com/PetervanLunteren/AddaxAI",
            },
            {
              label: "Addax Data Science",
              href: "https://addaxdatascience.com",
            },
            {
              label: "Contact",
              href: "mailto:peter@addaxdatascience.com",
            },
          ],
        },
      ],
      copyright: `Copyright © ${new Date().getFullYear()} Addax Data Science. Built with Docusaurus.`,
    },
    prism: {
      theme: prismThemes.github,
      darkTheme: prismThemes.dracula,
    },
  } satisfies Preset.ThemeConfig,
};

export default config;
