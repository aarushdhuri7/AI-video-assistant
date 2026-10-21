import ReactMarkdown from "react-markdown";
import remarkGfm from "remark-gfm";

/** Renders model-written Markdown. `skipHtml` drops any raw HTML the model might produce. */
export default function FormattedReport({ markdown }: { markdown: string }) {
  return (
    <div className="prose prose-zinc max-w-none prose-headings:font-semibold prose-h2:mt-6 prose-h2:text-lg prose-table:text-sm">
      <ReactMarkdown remarkPlugins={[remarkGfm]} skipHtml components={{ a: (p) => <a {...p} target="_blank" rel="noopener noreferrer" /> }}>
        {markdown}
      </ReactMarkdown>
    </div>
  );
}
