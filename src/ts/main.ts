import initPerformanceRnn from './performance_rnn';
import MuskiPerformanceRnnUi from "./muski-performance-rnn-ui";

$('[data-component="muski-performance-rnn-ui"]').each( (i, el) => {
  (async () => {
    const lang = $(el).data('lang') || 'en';
    const soundfontUrl = $(el).data('soundfont-url') || '/soundfonts/salamander-piano/';
    const checkpointUrl = $(el).data('checkpoint-url') || '/checkpoints/performance-rnn-tfjs';

    const $el = $(el);
    const originalChildren = $el.children();
    const ui = new MuskiPerformanceRnnUi({
      lang,
    });
    ui.$element.hide();
    $el.append(ui.$element);
    await initPerformanceRnn({
      soundfontUrl,
      checkpointUrl,
    });
    originalChildren.remove();
    ui.$element.show();
  })();
});
