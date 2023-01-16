import './performance_rnn';
import MuskiPerformanceRnnUi from "./muski-performance-rnn-ui";

$('[data-component="muski-performance-rnn-ui"]').each((i, el) => {
  const $el = $(el);
  const ui = new MuskiPerformanceRnnUi();
  $el.replaceWith(ui.$element);
});
